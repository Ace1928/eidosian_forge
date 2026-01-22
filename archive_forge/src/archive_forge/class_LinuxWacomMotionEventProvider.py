import os
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect
class LinuxWacomMotionEventProvider(MotionEventProvider):
    options = ('min_position_x', 'max_position_x', 'min_position_y', 'max_position_y', 'min_pressure', 'max_pressure', 'invert_x', 'invert_y')

    def __init__(self, device, args):
        super(LinuxWacomMotionEventProvider, self).__init__(device, args)
        self.input_fn = None
        self.default_ranges = dict()
        self.mode = 'touch'
        args = args.split(',')
        if not args:
            Logger.error('LinuxWacom: No filename given in config')
            Logger.error('LinuxWacom: Use /dev/input/event0 for example')
            return
        self.input_fn = args[0]
        Logger.info('LinuxWacom: Read event from <%s>' % self.input_fn)
        for arg in args[1:]:
            if arg == '':
                continue
            arg = arg.split('=')
            if len(arg) != 2:
                err = 'LinuxWacom: Bad parameter%s: Not in key=value format.' % arg
                Logger.error(err)
                continue
            key, value = arg
            if key == 'mode':
                self.mode = value
                continue
            if key not in LinuxWacomMotionEventProvider.options:
                Logger.error('LinuxWacom: unknown %s option' % key)
                continue
            try:
                self.default_ranges[key] = int(value)
            except ValueError:
                err = 'LinuxWacom: value %s invalid for %s' % (key, value)
                Logger.error(err)
                continue
            msg = 'LinuxWacom: Set custom %s to %d' % (key, int(value))
            Logger.info(msg)
        Logger.info('LinuxWacom: mode is <%s>' % self.mode)

    def start(self):
        if self.input_fn is None:
            return
        self.uid = 0
        self.queue = collections.deque()
        self.thread = threading.Thread(target=self._thread_run, kwargs=dict(queue=self.queue, input_fn=self.input_fn, device=self.device, default_ranges=self.default_ranges))
        self.thread.daemon = True
        self.thread.start()

    def _thread_run(self, **kwargs):
        input_fn = kwargs.get('input_fn')
        queue = kwargs.get('queue')
        device = kwargs.get('device')
        drs = kwargs.get('default_ranges').get
        touches = {}
        touches_sent = []
        l_points = {}
        range_min_position_x = 0
        range_max_position_x = 2048
        range_min_position_y = 0
        range_max_position_y = 2048
        range_min_pressure = 0
        range_max_pressure = 255
        invert_x = int(bool(drs('invert_x', 0)))
        invert_y = int(bool(drs('invert_y', 0)))
        reset_touch = False

        def process(points):
            actives = list(points.keys())
            for args in points.values():
                tid = args['id']
                try:
                    touch = touches[tid]
                except KeyError:
                    touch = LinuxWacomMotionEvent(device, tid, args)
                    touches[touch.id] = touch
                if touch.sx == args['x'] and touch.sy == args['y'] and (tid in touches_sent):
                    continue
                touch.move(args)
                if tid not in touches_sent:
                    queue.append(('begin', touch))
                    touches_sent.append(tid)
                queue.append(('update', touch))
            for tid in list(touches.keys())[:]:
                if tid not in actives:
                    touch = touches[tid]
                    if tid in touches_sent:
                        touch.update_time_end()
                        queue.append(('end', touch))
                        touches_sent.remove(tid)
                    del touches[tid]

        def normalize(value, vmin, vmax):
            return (value - vmin) / float(vmax - vmin)
        try:
            fd = open(input_fn, 'rb')
        except IOError:
            Logger.exception('Unable to open %s' % input_fn)
            return
        device_name = fcntl.ioctl(fd, EVIOCGNAME + (256 << 16), ' ' * 256).split('\x00')[0]
        Logger.info('LinuxWacom: using <%s>' % device_name)
        bit = fcntl.ioctl(fd, EVIOCGBIT + (EV_MAX << 16), ' ' * sz_l)
        bit, = struct.unpack('Q', bit)
        for x in range(EV_MAX):
            if x != EV_ABS:
                continue
            if bit & 1 << x == 0:
                continue
            sbit = fcntl.ioctl(fd, EVIOCGBIT + x + (KEY_MAX << 16), ' ' * sz_l)
            sbit, = struct.unpack('Q', sbit)
            for y in range(KEY_MAX):
                if sbit & 1 << y == 0:
                    continue
                absinfo = fcntl.ioctl(fd, EVIOCGABS + y + (struct_input_absinfo_sz << 16), ' ' * struct_input_absinfo_sz)
                abs_value, abs_min, abs_max, abs_fuzz, abs_flat, abs_res = struct.unpack('iiiiii', absinfo)
                if y == ABS_X:
                    range_min_position_x = drs('min_position_x', abs_min)
                    range_max_position_x = drs('max_position_x', abs_max)
                    Logger.info('LinuxWacom: ' + '<%s> range position X is %d - %d' % (device_name, abs_min, abs_max))
                elif y == ABS_Y:
                    range_min_position_y = drs('min_position_y', abs_min)
                    range_max_position_y = drs('max_position_y', abs_max)
                    Logger.info('LinuxWacom: ' + '<%s> range position Y is %d - %d' % (device_name, abs_min, abs_max))
                elif y == ABS_PRESSURE:
                    range_min_pressure = drs('min_pressure', abs_min)
                    range_max_pressure = drs('max_pressure', abs_max)
                    Logger.info('LinuxWacom: ' + '<%s> range pressure is %d - %d' % (device_name, abs_min, abs_max))
        changed = False
        touch_id = 0
        touch_x = 0
        touch_y = 0
        touch_pressure = 0
        while fd:
            data = fd.read(struct_input_event_sz)
            if len(data) < struct_input_event_sz:
                break
            for i in range(len(data) / struct_input_event_sz):
                ev = data[i * struct_input_event_sz:]
                tv_sec, tv_usec, ev_type, ev_code, ev_value = struct.unpack('LLHHi', ev[:struct_input_event_sz])
                if ev_type == EV_SYN and ev_code == SYN_REPORT:
                    if touch_id in l_points:
                        p = l_points[touch_id]
                    else:
                        p = dict()
                        l_points[touch_id] = p
                    p['id'] = touch_id
                    if not reset_touch:
                        p['x'] = touch_x
                        p['y'] = touch_y
                        p['pressure'] = touch_pressure
                    if self.mode == 'pen' and touch_pressure == 0 and (not reset_touch):
                        del l_points[touch_id]
                    if changed:
                        if 'x' not in p:
                            reset_touch = False
                            continue
                        process(l_points)
                        changed = False
                    if reset_touch:
                        l_points.clear()
                        reset_touch = False
                        process(l_points)
                elif ev_type == EV_MSC and ev_code == MSC_SERIAL:
                    touch_id = ev_value
                elif ev_type == EV_ABS and ev_code == ABS_X:
                    val = normalize(ev_value, range_min_position_x, range_max_position_x)
                    if invert_x:
                        val = 1.0 - val
                    touch_x = val
                    changed = True
                elif ev_type == EV_ABS and ev_code == ABS_Y:
                    val = 1.0 - normalize(ev_value, range_min_position_y, range_max_position_y)
                    if invert_y:
                        val = 1.0 - val
                    touch_y = val
                    changed = True
                elif ev_type == EV_ABS and ev_code == ABS_PRESSURE:
                    touch_pressure = normalize(ev_value, range_min_pressure, range_max_pressure)
                    changed = True
                elif ev_type == EV_ABS and ev_code == ABS_MISC:
                    if ev_value == 0:
                        reset_touch = True

    def update(self, dispatch_fn):
        try:
            while True:
                event_type, touch = self.queue.popleft()
                dispatch_fn(event_type, touch)
        except:
            pass