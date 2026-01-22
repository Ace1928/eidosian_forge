import pyui  # type: ignore[import-not-found]
def _guiUpdate(reactor, delay):
    pyui.draw()
    if pyui.update() == 0:
        pyui.quit()
        reactor.stop()
    else:
        reactor.callLater(delay, _guiUpdate, reactor, delay)