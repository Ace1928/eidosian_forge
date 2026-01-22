from __future__ import unicode_literals
from prompt_toolkit.filters import to_cli_filter
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Point, Screen, WritePosition
from prompt_toolkit.output import Output
from prompt_toolkit.styles import Style
from prompt_toolkit.token import Token
from prompt_toolkit.utils import is_windows
from six.moves import range
def _output_screen_diff(output, screen, current_pos, previous_screen=None, last_token=None, is_done=False, attrs_for_token=None, size=None, previous_width=0):
    """
    Render the diff between this screen and the previous screen.

    This takes two `Screen` instances. The one that represents the output like
    it was during the last rendering and one that represents the current
    output raster. Looking at these two `Screen` instances, this function will
    render the difference by calling the appropriate methods of the `Output`
    object that only paint the changes to the terminal.

    This is some performance-critical code which is heavily optimized.
    Don't change things without profiling first.

    :param current_pos: Current cursor position.
    :param last_token: `Token` instance that represents the output attributes of
            the last drawn character. (Color/attributes.)
    :param attrs_for_token: :class:`._TokenToAttrsCache` instance.
    :param width: The width of the terminal.
    :param prevous_width: The width of the terminal during the last rendering.
    """
    width, height = (size.columns, size.rows)
    last_token = [last_token]
    write = output.write
    write_raw = output.write_raw
    _output_set_attributes = output.set_attributes
    _output_reset_attributes = output.reset_attributes
    _output_cursor_forward = output.cursor_forward
    _output_cursor_up = output.cursor_up
    _output_cursor_backward = output.cursor_backward
    output.hide_cursor()

    def reset_attributes():
        """ Wrapper around Output.reset_attributes. """
        _output_reset_attributes()
        last_token[0] = None

    def move_cursor(new):
        """ Move cursor to this `new` point. Returns the given Point. """
        current_x, current_y = (current_pos.x, current_pos.y)
        if new.y > current_y:
            reset_attributes()
            write('\r\n' * (new.y - current_y))
            current_x = 0
            _output_cursor_forward(new.x)
            return new
        elif new.y < current_y:
            _output_cursor_up(current_y - new.y)
        if current_x >= width - 1:
            write('\r')
            _output_cursor_forward(new.x)
        elif new.x < current_x or current_x >= width - 1:
            _output_cursor_backward(current_x - new.x)
        elif new.x > current_x:
            _output_cursor_forward(new.x - current_x)
        return new

    def output_char(char):
        """
        Write the output of this character.
        """
        the_last_token = last_token[0]
        if the_last_token and the_last_token == char.token:
            write(char.char)
        else:
            _output_set_attributes(attrs_for_token[char.token])
            write(char.char)
            last_token[0] = char.token
    if not previous_screen:
        output.disable_autowrap()
        reset_attributes()
    if is_done or not previous_screen or previous_width != width:
        current_pos = move_cursor(Point(0, 0))
        reset_attributes()
        output.erase_down()
        previous_screen = Screen()
    current_height = min(screen.height, height)
    row_count = min(max(screen.height, previous_screen.height), height)
    c = 0
    for y in range(row_count):
        new_row = screen.data_buffer[y]
        previous_row = previous_screen.data_buffer[y]
        zero_width_escapes_row = screen.zero_width_escapes[y]
        new_max_line_len = min(width - 1, max(new_row.keys()) if new_row else 0)
        previous_max_line_len = min(width - 1, max(previous_row.keys()) if previous_row else 0)
        c = 0
        while c < new_max_line_len + 1:
            new_char = new_row[c]
            old_char = previous_row[c]
            char_width = new_char.width or 1
            if new_char.char != old_char.char or new_char.token != old_char.token:
                current_pos = move_cursor(Point(y=y, x=c))
                if c in zero_width_escapes_row:
                    write_raw(zero_width_escapes_row[c])
                output_char(new_char)
                current_pos = current_pos._replace(x=current_pos.x + char_width)
            c += char_width
        if previous_screen and new_max_line_len < previous_max_line_len:
            current_pos = move_cursor(Point(y=y, x=new_max_line_len + 1))
            reset_attributes()
            output.erase_end_of_line()
    if current_height > previous_screen.height:
        current_pos = move_cursor(Point(y=current_height - 1, x=0))
    if is_done:
        current_pos = move_cursor(Point(y=current_height, x=0))
        output.erase_down()
    else:
        current_pos = move_cursor(screen.cursor_position)
    if is_done:
        output.enable_autowrap()
    reset_attributes()
    if screen.show_cursor or is_done:
        output.show_cursor()
    return (current_pos, last_token[0])