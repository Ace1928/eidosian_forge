from Xlib import X
from Xlib.protocol import rq, structs
class QueryFont(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(47), rq.Pad(1), rq.RequestLength(), rq.Fontable('font'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Object('min_bounds', structs.CharInfo), rq.Pad(4), rq.Object('max_bounds', structs.CharInfo), rq.Pad(4), rq.Card16('min_char_or_byte2'), rq.Card16('max_char_or_byte2'), rq.Card16('default_char'), rq.LengthOf('properties', 2), rq.Card8('draw_direction'), rq.Card8('min_byte1'), rq.Card8('max_byte1'), rq.Card8('all_chars_exist'), rq.Int16('font_ascent'), rq.Int16('font_descent'), rq.LengthOf('char_infos', 4), rq.List('properties', structs.FontProp), rq.List('char_infos', structs.CharInfo))