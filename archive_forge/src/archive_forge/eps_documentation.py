import time
from ase.utils import writer
from ase.io.utils import PlottingVariables, make_patch_list
Encapsulated PostScript writer.

        show_unit_cell: int
            0: Don't show unit cell (default).  1: Show unit cell.
            2: Show unit cell and make sure all of it is visible.
        