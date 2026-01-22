from __future__ import annotations
import abc
import time
import typing
from .common import BaseScreen
class CF635Screen(CFLCDScreen):
    """
    Crystal Fontz 635 display

    20x4 character display + cursor
    no foreground/background colors or settings supported

    see CGROM for list of close unicode matches to characters available

    6 button input
    up, down, left, right, enter (check mark), exit (cross)
    """
    DISPLAY_SIZE = (20, 4)
    CGROM = '①②③④⑤⑥⑦⑧①②③④⑤⑥⑦⑧►◄⇑⇓«»↖↗↙↘▲▼↲^ˇ█ !"#¤%&\'()*+,-./0123456789:;<=>?¡ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÑÜ§¿abcdefghijklmnopqrstuvwxyzäöñüà⁰¹²³⁴⁵⁶⁷⁸⁹½¼±≥≤μ♪♫⑴♥♦⑵⌜⌟“”()αɛδ∞@£$¥èéùìòÇᴾØøʳÅå⌂¢ΦτλΩπΨΣθΞ♈ÆæßÉΓΛΠϒ_ÈÊêçğŞşİι~◊▇▆▄▃▁ƒ▉▋▌▍▏⑶◽▪↑→↓←ÁÍÓÚÝáíóúýÔôŮůČĔŘŠŽčĕřšž[\\]{|}'
    cursor_style = CFLCDScreen.CURSOR_INVERTING_BLINKING_BLOCK

    def __init__(self, device_path: str, baud: int=115200, repeat_delay: float=0.5, repeat_next: float=0.125, key_map: Iterable[str]=('up', 'down', 'left', 'right', 'enter', 'esc')):
        """
        device_path -- eg. '/dev/ttyUSB0'
        baud -- baud rate
        repeat_delay -- seconds to wait before starting to repeat keys
        repeat_next -- time between each repeated key
        key_map -- the keys to send for this device's buttons
        """
        super().__init__(device_path, baud)
        self.repeat_delay = repeat_delay
        self.repeat_next = repeat_next
        self.key_repeat = KeyRepeatSimulator(repeat_delay, repeat_next)
        self.key_map = tuple(key_map)
        self._last_command = None
        self._last_command_time = 0
        self._command_queue: list[tuple[int, bytearray]] = []
        self._screen_buf = None
        self._previous_canvas = None
        self._update_cursor = False

    def get_input_descriptors(self) -> list[int]:
        """
        return the fd from our serial device so we get called
        on input and responses
        """
        return [self._device.fd]

    def get_input_nonblocking(self) -> tuple[None, list[str], list[int]]:
        """
        Return a (next_input_timeout, keys_pressed, raw_keycodes)
        tuple.

        The protocol for our device requires waiting for acks between
        each command, so this method responds to those as well as key
        press and release events.

        Key repeat events are simulated here as the device doesn't send
        any for us.

        raw_keycodes are the bytes of messages we received, which might
        not seem to have any correspondence to keys_pressed.
        """
        data_input: list[str] = []
        raw_data_input: list[int] = []
        timeout = None
        packet = self._read_packet()
        while packet:
            command, data = packet
            if command == self.CMD_KEY_ACTIVITY and data:
                d0 = data[0]
                if 1 <= d0 <= 12:
                    release = d0 > 6
                    keycode = d0 - release * 6 - 1
                    key = self.key_map[keycode]
                    if release:
                        self.key_repeat.release(key)
                    else:
                        data_input.append(key)
                        self.key_repeat.press(key)
                    raw_data_input.append(d0)
            elif command & 192 == 64 and command & 63 == self._last_command:
                self._send_next_command()
            packet = self._read_packet()
        next_repeat = self.key_repeat.next_event()
        if next_repeat:
            timeout, key = next_repeat
            if not timeout:
                data_input.append(key)
                self.key_repeat.sent_event()
                timeout = None
        return (timeout, data_input, raw_data_input)

    def _send_next_command(self) -> None:
        """
        send out the next command in the queue
        """
        if not self._command_queue:
            self._last_command = None
            return
        command, data = self._command_queue.pop(0)
        self._send_packet(command, data)
        self._last_command = command
        self._last_command_time = time.time()

    def queue_command(self, command: int, data: bytearray) -> None:
        self._command_queue.append((command, data))
        if self._last_command is None:
            self._send_next_command()

    def draw_screen(self, size: tuple[int, int], canvas: Canvas) -> None:
        if size != self.DISPLAY_SIZE:
            raise ValueError(size)
        if self._screen_buf:
            osb = self._screen_buf
        else:
            osb = []
        sb = []
        for y, row in enumerate(canvas.content()):
            text = [run for _a, _cs, run in row]
            if not osb or osb[y] != text:
                data = bytearray([0, y])
                for elem in text:
                    data.extend(elem)
                self.queue_command(self.CMD_LCD_DATA, data)
            sb.append(text)
        if self._previous_canvas and self._previous_canvas.cursor == canvas.cursor and (not self._update_cursor or not canvas.cursor):
            pass
        elif canvas.cursor is None:
            self.queue_command(self.CMD_CURSOR_STYLE, bytearray([self.CURSOR_NONE]))
        else:
            x, y = canvas.cursor
            self.queue_command(self.CMD_CURSOR_POSITION, bytearray([x, y]))
            self.queue_command(self.CMD_CURSOR_STYLE, bytearray([self.cursor_style]))
        self._update_cursor = False
        self._screen_buf = sb
        self._previous_canvas = canvas

    def program_cgram(self, index: int, data: Sequence[int]) -> None:
        """
        Program character data.

        Characters available as chr(0) through chr(7), and repeated as chr(8) through chr(15).

        index -- 0 to 7 index of character to program

        data -- list of 8, 6-bit integer values top to bottom with MSB on the left side of the character.
        """
        if not 0 <= index <= 7:
            raise ValueError(index)
        if len(data) != 8:
            raise ValueError(data)
        self.queue_command(self.CMD_CGRAM, bytearray([index]) + bytearray(data))

    def set_cursor_style(self, style: Literal[1, 2, 3, 4]) -> None:
        """
        style -- CURSOR_BLINKING_BLOCK, CURSOR_UNDERSCORE,
            CURSOR_BLINKING_BLOCK_UNDERSCORE or
            CURSOR_INVERTING_BLINKING_BLOCK
        """
        if not 1 <= style <= 4:
            raise ValueError(style)
        self.cursor_style = style
        self._update_cursor = True

    def set_backlight(self, value: int) -> None:
        """
        Set backlight brightness

        value -- 0 to 100
        """
        if not 0 <= value <= 100:
            raise ValueError(value)
        self.queue_command(self.CMD_BACKLIGHT, bytearray([value]))

    def set_lcd_contrast(self, value: int) -> None:
        """
        value -- 0 to 255
        """
        if not 0 <= value <= 255:
            raise ValueError(value)
        self.queue_command(self.CMD_LCD_CONTRAST, bytearray([value]))

    def set_led_pin(self, led: Literal[0, 1, 2, 3], rg: Literal[0, 1], value: int) -> None:
        """
        led -- 0 to 3
        rg -- 0 for red, 1 for green
        value -- 0 to 100
        """
        if not 0 <= led <= 3:
            raise ValueError(led)
        if rg not in {0, 1}:
            raise ValueError(rg)
        if not 0 <= value <= 100:
            raise ValueError(value)
        self.queue_command(self.CMD_GPO, bytearray([12 - 2 * led - rg, value]))