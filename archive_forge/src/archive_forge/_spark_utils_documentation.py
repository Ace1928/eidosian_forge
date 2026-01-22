import contextlib
import multiprocessing
import os
import shutil
import tempfile
import zipfile
Given a path returned by add_local_model(), this method will return a tuple of
        (loaded_model, local_model_path).
        If this Python process ever loaded the model before, we will reuse that copy.
        