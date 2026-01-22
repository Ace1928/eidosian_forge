import signal
from pyglet import app
from pyglet.app.base import PlatformEventLoop, EventLoop
from pyglet.libs.darwin import cocoapy, AutoReleasePool, ObjCSubclass, PyObjectEncoding, ObjCInstance, send_super, \
class CocoaPlatformEventLoop(PlatformEventLoop):

    def __init__(self):
        super(CocoaPlatformEventLoop, self).__init__()
        with AutoReleasePool():
            self.NSApp = NSApplication.sharedApplication()
            if self.NSApp.isRunning():
                return
            if not self.NSApp.mainMenu():
                create_menu()
            self.NSApp.setActivationPolicy_(cocoapy.NSApplicationActivationPolicyRegular)
            defaults = NSUserDefaults.standardUserDefaults()
            ignoreState = cocoapy.CFSTR('ApplePersistenceIgnoreState')
            if not defaults.objectForKey_(ignoreState):
                defaults.setBool_forKey_(True, ignoreState)
            holdEnabled = cocoapy.CFSTR('ApplePressAndHoldEnabled')
            if not defaults.objectForKey_(holdEnabled):
                defaults.setBool_forKey_(False, holdEnabled)
            self._finished_launching = False

    def start(self):
        with AutoReleasePool():
            if not self.NSApp.isRunning() and (not self._finished_launching):
                self.NSApp.finishLaunching()
                self.NSApp.activateIgnoringOtherApps_(True)
                self._finished_launching = True

    def nsapp_start(self, interval):
        """Used only for CocoaAlternateEventLoop"""
        from pyglet.app import event_loop
        self._event_loop = event_loop

        def term_received(*args):
            if self.timer:
                self.timer.invalidate()
                self.timer = None
            self.nsapp_stop()
        signal.signal(signal.SIGINT, term_received)
        signal.signal(signal.SIGTERM, term_received)
        self.appdelegate = _AppDelegate.alloc().init(self)
        self.NSApp.setDelegate_(self.appdelegate)
        self.timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(interval, self.appdelegate, get_selector('updatePyglet:'), False, True)
        self.NSApp.run()

    def nsapp_step(self):
        """Used only for CocoaAlternateEventLoop"""
        self._event_loop.idle()
        self.dispatch_posted_events()

    def nsapp_stop(self):
        """Used only for CocoaAlternateEventLoop"""
        self.NSApp.terminate_(None)

    def step(self, timeout=None):
        with AutoReleasePool():
            self.dispatch_posted_events()
            if timeout is None:
                timeout_date = NSDate.distantFuture()
            elif timeout == 0.0:
                timeout_date = NSDate.distantPast()
            else:
                timeout_date = NSDate.dateWithTimeIntervalSinceNow_(timeout)
            self._is_running.set()
            event = self.NSApp.nextEventMatchingMask_untilDate_inMode_dequeue_(cocoapy.NSAnyEventMask, timeout_date, cocoapy.NSDefaultRunLoopMode, True)
            if event is not None:
                event_type = event.type()
                if event_type != cocoapy.NSApplicationDefined:
                    self.NSApp.sendEvent_(event)
                self.NSApp.updateWindows()
                did_time_out = False
            else:
                did_time_out = True
            self._is_running.clear()
            return did_time_out

    def stop(self):
        pass

    def notify(self):
        with AutoReleasePool():
            notifyEvent = NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(cocoapy.NSApplicationDefined, cocoapy.NSPoint(0.0, 0.0), 0, 0, 0, None, 0, 0, 0)
            self.NSApp.postEvent_atStart_(notifyEvent, False)