import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
Calculates the installation directory.

    ~/.cupy/cuda_lib/{cuda_version}/{library_name}/{library_version}
    