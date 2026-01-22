from __future__ import absolute_import
import codecs
import os
import sys
import threading
import serial
from serial.tools.list_ports import comports
from serial.tools import hexlify_codec
class Miniterm(object):
    """    Terminal application. Copy data from serial port to console and vice versa.
    Handle special keys from the console to show menu etc.
    """

    def __init__(self, serial_instance, echo=False, eol='crlf', filters=()):
        self.console = Console()
        self.serial = serial_instance
        self.echo = echo
        self.raw = False
        self.input_encoding = 'UTF-8'
        self.output_encoding = 'UTF-8'
        self.eol = eol
        self.filters = filters
        self.update_transformations()
        self.exit_character = unichr(29)
        self.menu_character = unichr(20)
        self.alive = None
        self._reader_alive = None
        self.receiver_thread = None
        self.rx_decoder = None
        self.tx_decoder = None

    def _start_reader(self):
        """Start reader thread"""
        self._reader_alive = True
        self.receiver_thread = threading.Thread(target=self.reader, name='rx')
        self.receiver_thread.daemon = True
        self.receiver_thread.start()

    def _stop_reader(self):
        """Stop reader thread only, wait for clean exit of thread"""
        self._reader_alive = False
        if hasattr(self.serial, 'cancel_read'):
            self.serial.cancel_read()
        self.receiver_thread.join()

    def start(self):
        """start worker threads"""
        self.alive = True
        self._start_reader()
        self.transmitter_thread = threading.Thread(target=self.writer, name='tx')
        self.transmitter_thread.daemon = True
        self.transmitter_thread.start()
        self.console.setup()

    def stop(self):
        """set flag to stop worker threads"""
        self.alive = False

    def join(self, transmit_only=False):
        """wait for worker threads to terminate"""
        self.transmitter_thread.join()
        if not transmit_only:
            if hasattr(self.serial, 'cancel_read'):
                self.serial.cancel_read()
            self.receiver_thread.join()

    def close(self):
        self.serial.close()

    def update_transformations(self):
        """take list of transformation classes and instantiate them for rx and tx"""
        transformations = [EOL_TRANSFORMATIONS[self.eol]] + [TRANSFORMATIONS[f] for f in self.filters]
        self.tx_transformations = [t() for t in transformations]
        self.rx_transformations = list(reversed(self.tx_transformations))

    def set_rx_encoding(self, encoding, errors='replace'):
        """set encoding for received data"""
        self.input_encoding = encoding
        self.rx_decoder = codecs.getincrementaldecoder(encoding)(errors)

    def set_tx_encoding(self, encoding, errors='replace'):
        """set encoding for transmitted data"""
        self.output_encoding = encoding
        self.tx_encoder = codecs.getincrementalencoder(encoding)(errors)

    def dump_port_settings(self):
        """Write current settings to sys.stderr"""
        sys.stderr.write('\n--- Settings: {p.name}  {p.baudrate},{p.bytesize},{p.parity},{p.stopbits}\n'.format(p=self.serial))
        sys.stderr.write('--- RTS: {:8}  DTR: {:8}  BREAK: {:8}\n'.format('active' if self.serial.rts else 'inactive', 'active' if self.serial.dtr else 'inactive', 'active' if self.serial.break_condition else 'inactive'))
        try:
            sys.stderr.write('--- CTS: {:8}  DSR: {:8}  RI: {:8}  CD: {:8}\n'.format('active' if self.serial.cts else 'inactive', 'active' if self.serial.dsr else 'inactive', 'active' if self.serial.ri else 'inactive', 'active' if self.serial.cd else 'inactive'))
        except serial.SerialException:
            pass
        sys.stderr.write('--- software flow control: {}\n'.format('active' if self.serial.xonxoff else 'inactive'))
        sys.stderr.write('--- hardware flow control: {}\n'.format('active' if self.serial.rtscts else 'inactive'))
        sys.stderr.write('--- serial input encoding: {}\n'.format(self.input_encoding))
        sys.stderr.write('--- serial output encoding: {}\n'.format(self.output_encoding))
        sys.stderr.write('--- EOL: {}\n'.format(self.eol.upper()))
        sys.stderr.write('--- filters: {}\n'.format(' '.join(self.filters)))

    def reader(self):
        """loop and copy serial->console"""
        try:
            while self.alive and self._reader_alive:
                data = self.serial.read(self.serial.in_waiting or 1)
                if data:
                    if self.raw:
                        self.console.write_bytes(data)
                    else:
                        text = self.rx_decoder.decode(data)
                        for transformation in self.rx_transformations:
                            text = transformation.rx(text)
                        self.console.write(text)
        except serial.SerialException:
            self.alive = False
            self.console.cancel()
            raise

    def writer(self):
        """        Loop and copy console->serial until self.exit_character character is
        found. When self.menu_character is found, interpret the next key
        locally.
        """
        menu_active = False
        try:
            while self.alive:
                try:
                    c = self.console.getkey()
                except KeyboardInterrupt:
                    c = '\x03'
                if not self.alive:
                    break
                if menu_active:
                    self.handle_menu_key(c)
                    menu_active = False
                elif c == self.menu_character:
                    menu_active = True
                elif c == self.exit_character:
                    self.stop()
                    break
                else:
                    text = c
                    for transformation in self.tx_transformations:
                        text = transformation.tx(text)
                    self.serial.write(self.tx_encoder.encode(text))
                    if self.echo:
                        echo_text = c
                        for transformation in self.tx_transformations:
                            echo_text = transformation.echo(echo_text)
                        self.console.write(echo_text)
        except:
            self.alive = False
            raise

    def handle_menu_key(self, c):
        """Implement a simple menu / settings"""
        if c == self.menu_character or c == self.exit_character:
            self.serial.write(self.tx_encoder.encode(c))
            if self.echo:
                self.console.write(c)
        elif c == '\x15':
            self.upload_file()
        elif c in '\x08hH?':
            sys.stderr.write(self.get_help_text())
        elif c == '\x12':
            self.serial.rts = not self.serial.rts
            sys.stderr.write('--- RTS {} ---\n'.format('active' if self.serial.rts else 'inactive'))
        elif c == '\x04':
            self.serial.dtr = not self.serial.dtr
            sys.stderr.write('--- DTR {} ---\n'.format('active' if self.serial.dtr else 'inactive'))
        elif c == '\x02':
            self.serial.break_condition = not self.serial.break_condition
            sys.stderr.write('--- BREAK {} ---\n'.format('active' if self.serial.break_condition else 'inactive'))
        elif c == '\x05':
            self.echo = not self.echo
            sys.stderr.write('--- local echo {} ---\n'.format('active' if self.echo else 'inactive'))
        elif c == '\x06':
            self.change_filter()
        elif c == '\x0c':
            modes = list(EOL_TRANSFORMATIONS)
            eol = modes.index(self.eol) + 1
            if eol >= len(modes):
                eol = 0
            self.eol = modes[eol]
            sys.stderr.write('--- EOL: {} ---\n'.format(self.eol.upper()))
            self.update_transformations()
        elif c == '\x01':
            self.change_encoding()
        elif c == '\t':
            self.dump_port_settings()
        elif c in 'pP':
            self.change_port()
        elif c in 'zZ':
            self.suspend_port()
        elif c in 'bB':
            self.change_baudrate()
        elif c == '8':
            self.serial.bytesize = serial.EIGHTBITS
            self.dump_port_settings()
        elif c == '7':
            self.serial.bytesize = serial.SEVENBITS
            self.dump_port_settings()
        elif c in 'eE':
            self.serial.parity = serial.PARITY_EVEN
            self.dump_port_settings()
        elif c in 'oO':
            self.serial.parity = serial.PARITY_ODD
            self.dump_port_settings()
        elif c in 'mM':
            self.serial.parity = serial.PARITY_MARK
            self.dump_port_settings()
        elif c in 'sS':
            self.serial.parity = serial.PARITY_SPACE
            self.dump_port_settings()
        elif c in 'nN':
            self.serial.parity = serial.PARITY_NONE
            self.dump_port_settings()
        elif c == '1':
            self.serial.stopbits = serial.STOPBITS_ONE
            self.dump_port_settings()
        elif c == '2':
            self.serial.stopbits = serial.STOPBITS_TWO
            self.dump_port_settings()
        elif c == '3':
            self.serial.stopbits = serial.STOPBITS_ONE_POINT_FIVE
            self.dump_port_settings()
        elif c in 'xX':
            self.serial.xonxoff = c == 'X'
            self.dump_port_settings()
        elif c in 'rR':
            self.serial.rtscts = c == 'R'
            self.dump_port_settings()
        elif c in 'qQ':
            self.stop()
        else:
            sys.stderr.write('--- unknown menu character {} --\n'.format(key_description(c)))

    def upload_file(self):
        """Ask user for filenname and send its contents"""
        sys.stderr.write('\n--- File to upload: ')
        sys.stderr.flush()
        with self.console:
            filename = sys.stdin.readline().rstrip('\r\n')
            if filename:
                try:
                    with open(filename, 'rb') as f:
                        sys.stderr.write('--- Sending file {} ---\n'.format(filename))
                        while True:
                            block = f.read(1024)
                            if not block:
                                break
                            self.serial.write(block)
                            self.serial.flush()
                            sys.stderr.write('.')
                    sys.stderr.write('\n--- File {} sent ---\n'.format(filename))
                except IOError as e:
                    sys.stderr.write('--- ERROR opening file {}: {} ---\n'.format(filename, e))

    def change_filter(self):
        """change the i/o transformations"""
        sys.stderr.write('\n--- Available Filters:\n')
        sys.stderr.write('\n'.join(('---   {:<10} = {.__doc__}'.format(k, v) for k, v in sorted(TRANSFORMATIONS.items()))))
        sys.stderr.write('\n--- Enter new filter name(s) [{}]: '.format(' '.join(self.filters)))
        with self.console:
            new_filters = sys.stdin.readline().lower().split()
        if new_filters:
            for f in new_filters:
                if f not in TRANSFORMATIONS:
                    sys.stderr.write('--- unknown filter: {!r}\n'.format(f))
                    break
            else:
                self.filters = new_filters
                self.update_transformations()
        sys.stderr.write('--- filters: {}\n'.format(' '.join(self.filters)))

    def change_encoding(self):
        """change encoding on the serial port"""
        sys.stderr.write('\n--- Enter new encoding name [{}]: '.format(self.input_encoding))
        with self.console:
            new_encoding = sys.stdin.readline().strip()
        if new_encoding:
            try:
                codecs.lookup(new_encoding)
            except LookupError:
                sys.stderr.write('--- invalid encoding name: {}\n'.format(new_encoding))
            else:
                self.set_rx_encoding(new_encoding)
                self.set_tx_encoding(new_encoding)
        sys.stderr.write('--- serial input encoding: {}\n'.format(self.input_encoding))
        sys.stderr.write('--- serial output encoding: {}\n'.format(self.output_encoding))

    def change_baudrate(self):
        """change the baudrate"""
        sys.stderr.write('\n--- Baudrate: ')
        sys.stderr.flush()
        with self.console:
            backup = self.serial.baudrate
            try:
                self.serial.baudrate = int(sys.stdin.readline().strip())
            except ValueError as e:
                sys.stderr.write('--- ERROR setting baudrate: {} ---\n'.format(e))
                self.serial.baudrate = backup
            else:
                self.dump_port_settings()

    def change_port(self):
        """Have a conversation with the user to change the serial port"""
        with self.console:
            try:
                port = ask_for_port()
            except KeyboardInterrupt:
                port = None
        if port and port != self.serial.port:
            self._stop_reader()
            settings = self.serial.getSettingsDict()
            try:
                new_serial = serial.serial_for_url(port, do_not_open=True)
                new_serial.applySettingsDict(settings)
                new_serial.rts = self.serial.rts
                new_serial.dtr = self.serial.dtr
                new_serial.open()
                new_serial.break_condition = self.serial.break_condition
            except Exception as e:
                sys.stderr.write('--- ERROR opening new port: {} ---\n'.format(e))
                new_serial.close()
            else:
                self.serial.close()
                self.serial = new_serial
                sys.stderr.write('--- Port changed to: {} ---\n'.format(self.serial.port))
            self._start_reader()

    def suspend_port(self):
        """        open port temporarily, allow reconnect, exit and port change to get
        out of the loop
        """
        self._stop_reader()
        self.serial.close()
        sys.stderr.write('\n--- Port closed: {} ---\n'.format(self.serial.port))
        do_change_port = False
        while not self.serial.is_open:
            sys.stderr.write('--- Quit: {exit} | p: port change | any other key to reconnect ---\n'.format(exit=key_description(self.exit_character)))
            k = self.console.getkey()
            if k == self.exit_character:
                self.stop()
                break
            elif k in 'pP':
                do_change_port = True
                break
            try:
                self.serial.open()
            except Exception as e:
                sys.stderr.write('--- ERROR opening port: {} ---\n'.format(e))
        if do_change_port:
            self.change_port()
        else:
            self._start_reader()
            sys.stderr.write('--- Port opened: {} ---\n'.format(self.serial.port))

    def get_help_text(self):
        """return the help text"""
        return '\n--- pySerial ({version}) - miniterm - help\n---\n--- {exit:8} Exit program (alias {menu} Q)\n--- {menu:8} Menu escape key, followed by:\n--- Menu keys:\n---    {menu:7} Send the menu character itself to remote\n---    {exit:7} Send the exit character itself to remote\n---    {info:7} Show info\n---    {upload:7} Upload file (prompt will be shown)\n---    {repr:7} encoding\n---    {filter:7} edit filters\n--- Toggles:\n---    {rts:7} RTS   {dtr:7} DTR   {brk:7} BREAK\n---    {echo:7} echo  {eol:7} EOL\n---\n--- Port settings ({menu} followed by the following):\n---    p          change port\n---    7 8        set data bits\n---    N E O S M  change parity (None, Even, Odd, Space, Mark)\n---    1 2 3      set stop bits (1, 2, 1.5)\n---    b          change baud rate\n---    x X        disable/enable software flow control\n---    r R        disable/enable hardware flow control\n'.format(version=getattr(serial, 'VERSION', 'unknown version'), exit=key_description(self.exit_character), menu=key_description(self.menu_character), rts=key_description('\x12'), dtr=key_description('\x04'), brk=key_description('\x02'), echo=key_description('\x05'), info=key_description('\t'), upload=key_description('\x15'), repr=key_description('\x01'), filter=key_description('\x06'), eol=key_description('\x0c'))