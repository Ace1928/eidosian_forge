import wx
def OnReplSelect(self, evt):
    """Callback when a new replacement option is selected."""
    sel = self.replace_list.GetSelection()
    if sel == -1:
        return
    opt = self.replace_list.GetString(sel)
    self.replace_text.SetValue(opt)