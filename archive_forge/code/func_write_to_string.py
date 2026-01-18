from ._LinearDrawer import LinearDrawer
from ._CircularDrawer import CircularDrawer
from ._Track import Track
from Bio.Graphics import _write
def write_to_string(self, output='PS', dpi=72):
    """Return a byte string containing the diagram in the requested format.

        Arguments:
            - output    - a string indicating output format, one of PS, PDF,
              SVG, JPG, BMP, GIF, PNG, TIFF or TIFF (as specified for the write
              method).
            - dpi       - Resolution (dots per inch) for bitmap formats.

        Returns:
            Return the completed drawing as a bytes string in a prescribed
            format.

        """
    from io import BytesIO
    handle = BytesIO()
    self.write(handle, output, dpi)
    return handle.getvalue()