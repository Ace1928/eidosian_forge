from traitlets import Bool, Set
from .base import Preprocessor

        Preprocessing to apply on each notebook.

        Must return modified nb, resources.

        Parameters
        ----------
        nb : NotebookNode
            Notebook being converted
        resources : dictionary
            Additional resources used in the conversion process.  Allows
            preprocessors to pass variables into the Jinja engine.
        