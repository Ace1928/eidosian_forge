import pdb
import bpython
class BPdb(pdb.Pdb):
    """PDB with BPython support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt = '(BPdb) '
        self.intro = 'Use "B" to enter bpython, Ctrl-d to exit it.'

    def postloop(self):
        self.intro = None
        super().postloop()

    def do_Bpython(self, arg):
        locals_ = self.curframe.f_globals.copy()
        locals_.update(self.curframe.f_locals)
        bpython.embed(locals_, ['-i'])

    def help_Bpython(self):
        print('B(python)')
        print('')
        print('Invoke the bpython interpreter for this stack frame. To exit bpython and return to a standard pdb press Ctrl-d')
    do_B = do_Bpython
    help_B = help_Bpython