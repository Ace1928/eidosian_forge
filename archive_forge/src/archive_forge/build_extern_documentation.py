import argparse
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

        Invoke llvm-dis to disassemble the given file.
        :param path: path to llvm-dis
        