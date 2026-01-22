import wx
class TestDialog(wxSpellCheckerDialog):

    def __init__(self, *args):
        super().__init__(*args)
        wx.EVT_CLOSE(self, self.OnClose)

    def OnClose(self, evnt):
        chkr = dlg.GetSpellChecker()
        if chkr is not None:
            print(['AFTER:', chkr.get_text()])
        self.Destroy()