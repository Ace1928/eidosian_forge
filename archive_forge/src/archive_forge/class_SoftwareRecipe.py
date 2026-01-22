from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SoftwareRecipe(_messages.Message):
    """A software recipe is a set of instructions for installing and
  configuring a piece of software. It consists of a set of artifacts that are
  downloaded, and a set of steps that install, configure, and/or update the
  software. Recipes support installing and updating software from artifacts in
  the following formats: Zip archive, Tar archive, Windows MSI, Debian
  package, and RPM package. Additionally, recipes support executing a script
  (either defined in a file or directly in this api) in bash, sh, cmd, and
  powershell. Updating a software recipe If a recipe is assigned to an
  instance and there is a recipe with the same name but a lower version
  already installed and the assigned state of the recipe is `UPDATED`, then
  the recipe is updated to the new version. Script Working Directories Each
  script or execution step is run in its own temporary directory which is
  deleted after completing the step.

  Enums:
    DesiredStateValueValuesEnum: Default is INSTALLED. The desired state the
      agent should maintain for this recipe. INSTALLED: The software recipe is
      installed on the instance but won't be updated to new versions. UPDATED:
      The software recipe is installed on the instance. The recipe is updated
      to a higher version, if a higher version of the recipe is assigned to
      this instance. REMOVE: Remove is unsupported for software recipes and
      attempts to create or update a recipe to the REMOVE state is rejected.

  Fields:
    artifacts: Resources available to be used in the steps in the recipe.
    desiredState: Default is INSTALLED. The desired state the agent should
      maintain for this recipe. INSTALLED: The software recipe is installed on
      the instance but won't be updated to new versions. UPDATED: The software
      recipe is installed on the instance. The recipe is updated to a higher
      version, if a higher version of the recipe is assigned to this instance.
      REMOVE: Remove is unsupported for software recipes and attempts to
      create or update a recipe to the REMOVE state is rejected.
    installSteps: Actions to be taken for installing this recipe. On failure
      it stops executing steps and does not attempt another installation. Any
      steps taken (including partially completed steps) are not rolled back.
    name: Required. Unique identifier for the recipe. Only one recipe with a
      given name is installed on an instance. Names are also used to identify
      resources which helps to determine whether guest policies have
      conflicts. This means that requests to create multiple recipes with the
      same name and version are rejected since they could potentially have
      conflicting assignments.
    updateSteps: Actions to be taken for updating this recipe. On failure it
      stops executing steps and does not attempt another update for this
      recipe. Any steps taken (including partially completed steps) are not
      rolled back.
    version: The version of this software recipe. Version can be up to 4
      period separated numbers (e.g. 12.34.56.78).
  """

    class DesiredStateValueValuesEnum(_messages.Enum):
        """Default is INSTALLED. The desired state the agent should maintain for
    this recipe. INSTALLED: The software recipe is installed on the instance
    but won't be updated to new versions. UPDATED: The software recipe is
    installed on the instance. The recipe is updated to a higher version, if a
    higher version of the recipe is assigned to this instance. REMOVE: Remove
    is unsupported for software recipes and attempts to create or update a
    recipe to the REMOVE state is rejected.

    Values:
      DESIRED_STATE_UNSPECIFIED: The default is to ensure the package is
        installed.
      INSTALLED: The agent ensures that the package is installed.
      UPDATED: The agent ensures that the package is installed and
        periodically checks for and install any updates.
      REMOVED: The agent ensures that the package is not installed and
        uninstall it if detected.
    """
        DESIRED_STATE_UNSPECIFIED = 0
        INSTALLED = 1
        UPDATED = 2
        REMOVED = 3
    artifacts = _messages.MessageField('SoftwareRecipeArtifact', 1, repeated=True)
    desiredState = _messages.EnumField('DesiredStateValueValuesEnum', 2)
    installSteps = _messages.MessageField('SoftwareRecipeStep', 3, repeated=True)
    name = _messages.StringField(4)
    updateSteps = _messages.MessageField('SoftwareRecipeStep', 5, repeated=True)
    version = _messages.StringField(6)