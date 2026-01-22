from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class EffectiveGuestPolicy(_messages.Message):
    """The effective guest policy that applies to a VM instance.

  Fields:
    packageRepositories: List of package repository configurations assigned to
      the VM instance.
    packages: List of package configurations assigned to the VM instance.
    softwareRecipes: List of recipes assigned to the VM instance.
  """
    packageRepositories = _messages.MessageField('EffectiveGuestPolicySourcedPackageRepository', 1, repeated=True)
    packages = _messages.MessageField('EffectiveGuestPolicySourcedPackage', 2, repeated=True)
    softwareRecipes = _messages.MessageField('EffectiveGuestPolicySourcedSoftwareRecipe', 3, repeated=True)