from matplotlib import _api, backend_tools, cbook, widgets
def _handle_toggle(self, tool, canvasevent, data):
    """
        Toggle tools, need to untoggle prior to using other Toggle tool.
        Called from trigger_tool.

        Parameters
        ----------
        tool : `.ToolBase`
        canvasevent : Event
            Original Canvas event or None.
        data : object
            Extra data to pass to the tool when triggering.
        """
    radio_group = tool.radio_group
    if radio_group is None:
        if tool.name in self._toggled[None]:
            self._toggled[None].remove(tool.name)
        else:
            self._toggled[None].add(tool.name)
        return
    if self._toggled[radio_group] == tool.name:
        toggled = None
    elif self._toggled[radio_group] is None:
        toggled = tool.name
    else:
        self.trigger_tool(self._toggled[radio_group], self, canvasevent, data)
        toggled = tool.name
    self._toggled[radio_group] = toggled