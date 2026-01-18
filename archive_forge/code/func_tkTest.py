import math
from string import ascii_letters as LETTERS
from rdkit.sping import pagesizes
from rdkit.sping.pid import *
def tkTest(testfunc):
    try:
        import sping.TK
        import Tkinter
    except ImportError:
        print('A module needed for sping.TK is not available, select another backend')
        return
    root = Tkinter.Tk()
    frame = Tkinter.Frame(root)
    tkcanvas = sping.TK.TKCanvas(size=(400, 400), name='sping-testTK', master=frame)
    bframe = Tkinter.Frame(root)
    minimalB = Tkinter.Button(bframe, text='minimal test', command=lambda c=tkcanvas: (c.clear(), drawMinimal(c), c.flush())).pack(side=Tkinter.LEFT)
    basicB = Tkinter.Button(bframe, text='basic test', command=lambda c=tkcanvas: (c.clear(), drawBasics(c), c.flush())).pack(side=Tkinter.LEFT)
    spectB = Tkinter.Button(bframe, text='spectrum test', command=lambda c=tkcanvas: (c.clear(), drawSpectrum(c), c.flush())).pack(side=Tkinter.LEFT)
    stringsB = Tkinter.Button(bframe, text='strings test', command=lambda c=tkcanvas: (c.clear(), drawStrings(c), c.flush())).pack(side=Tkinter.LEFT)
    rotstrB = Tkinter.Button(bframe, text='rotated strings test', command=lambda c=tkcanvas: (c.clear(), drawRotstring(c), c.flush())).pack(side=Tkinter.LEFT)
    advancedB = Tkinter.Button(bframe, text='advanced test', command=lambda c=tkcanvas: (c.clear(), drawAdvanced(c), c.flush())).pack(side=Tkinter.LEFT)
    bframe.pack(side=Tkinter.TOP)
    tkcanvas.pack()
    frame.pack()
    if testfunc == minimal:
        drawMinimal(tkcanvas)
    elif testfunc == basics:
        drawBasics(tkcanvas)
    elif testfunc == advanced:
        drawAdvanced(tkcanvas)
    elif testfunc == spectrum:
        drawSpectrum(tkcanvas)
    elif testfunc == strings:
        drawStrings(tkcanvas)
    elif testfunc == rotstring:
        drawRotstring(tkcanvas)
    else:
        print('Illegal testfunc handed to tkTest')
        raise ValueError('Unsupported testfunc')
    tkcanvas.flush()
    root.mainloop()