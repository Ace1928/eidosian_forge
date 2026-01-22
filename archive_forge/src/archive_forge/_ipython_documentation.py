import os
import shutil
import subprocess
import tempfile
from typing import Any, Optional
from IPython.display import Image
from pyquil.latex._main import to_latex
from pyquil.latex._diagram import DiagramSettings
from pyquil.quil import Program

    Displays a PyQuil circuit as an IPython image object.

    .. note::

       For this to work, you need two external programs, ``pdflatex`` and ``convert``,
       to be installed and accessible via your shell path.

       Further, your LaTeX installation should include class and style files for ``standalone``,
       ``geometry``, ``tikz``, and ``quantikz``. If it does not, you need to install
       these yourself.

    :param circuit: The circuit to be drawn, represented as a pyquil program.
    :param settings: An optional object of settings controlling diagram rendering and layout.
    :return: PNG image render of the circuit.
    