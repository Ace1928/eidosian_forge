from ase import Atoms
def view_ngl(atoms, w=500, h=500):
    """
    Returns the nglviewer + some control widgets in the VBox ipywidget.
    The viewer supports any Atoms objectand any sequence of Atoms objects.
    The returned object has two shortcuts members:

    .view:
        nglviewer ipywidget for direct interaction
    .control_box:
        VBox ipywidget containing view control widgets
    """
    return NGLDisplay(atoms, w, h).gui