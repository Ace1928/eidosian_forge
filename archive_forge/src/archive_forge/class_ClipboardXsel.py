from kivy.utils import platform
from kivy.core.clipboard._clipboard_ext import ClipboardExternalBase
class ClipboardXsel(ClipboardExternalBase):

    @staticmethod
    def _clip(inout, selection):
        pipe = {'std' + inout: subprocess.PIPE}
        sel = 'b' if selection == 'clipboard' else selection[0]
        io = inout[0]
        return subprocess.Popen(['xsel', '-' + sel + io], **pipe)