from winappdbg import win32
def __get_window(self, hWnd):
    """
        User internally to get another Window from this one.
        It'll try to copy the parent Process and Thread references if possible.
        """
    window = Window(hWnd)
    if window.get_pid() == self.get_pid():
        window.set_process(self.get_process())
    if window.get_tid() == self.get_tid():
        window.set_thread(self.get_thread())
    return window