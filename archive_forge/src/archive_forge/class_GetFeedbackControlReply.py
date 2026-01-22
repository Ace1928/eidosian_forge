import xcffib
import struct
import io
from . import xfixes
from . import xproto
class GetFeedbackControlReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.xi_reply_type, self.num_feedbacks = unpacker.unpack('xB2x4xH22x')
        self.feedbacks = xcffib.List(unpacker, FeedbackState, self.num_feedbacks)
        self.bufsize = unpacker.offset - base