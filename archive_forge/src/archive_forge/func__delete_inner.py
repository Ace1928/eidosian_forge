import os
import posixpath
import sys
import threading
import urllib.parse
from contextlib import contextmanager
from mlflow.entities import FileInfo
from mlflow.store.artifact.artifact_repo import ArtifactRepository
def _delete_inner(self, artifact_path, sftp):
    if sftp.isdir(artifact_path):
        with sftp.cd(artifact_path):
            for element in sftp.listdir():
                self._delete_inner(element, sftp)
        sftp.rmdir(artifact_path)
    elif sftp.isfile(artifact_path):
        sftp.remove(artifact_path)