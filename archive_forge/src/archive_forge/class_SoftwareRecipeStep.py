from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SoftwareRecipeStep(_messages.Message):
    """An action that can be taken as part of installing or updating a recipe.

  Fields:
    archiveExtraction: Extracts an archive into the specified directory.
    dpkgInstallation: Installs a deb file via dpkg.
    fileCopy: Copies a file onto the instance.
    fileExec: Executes an artifact or local file.
    msiInstallation: Installs an MSI file.
    rpmInstallation: Installs an rpm file via the rpm utility.
    scriptRun: Runs commands in a shell.
  """
    archiveExtraction = _messages.MessageField('SoftwareRecipeStepExtractArchive', 1)
    dpkgInstallation = _messages.MessageField('SoftwareRecipeStepInstallDpkg', 2)
    fileCopy = _messages.MessageField('SoftwareRecipeStepCopyFile', 3)
    fileExec = _messages.MessageField('SoftwareRecipeStepExecFile', 4)
    msiInstallation = _messages.MessageField('SoftwareRecipeStepInstallMsi', 5)
    rpmInstallation = _messages.MessageField('SoftwareRecipeStepInstallRpm', 6)
    scriptRun = _messages.MessageField('SoftwareRecipeStepRunScript', 7)