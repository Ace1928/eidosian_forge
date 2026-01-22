from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GuestPolicy(_messages.Message):
    """An OS Config resource representing a guest configuration policy. These
  policies represent the desired state for VM instance guest environments
  including packages to install or remove, package repository configurations,
  and software to install.

  Fields:
    assignment: Required. Specifies the VM instances that are assigned to this
      policy. This allows you to target sets or groups of VM instances by
      different parameters such as labels, names, OS, or zones. If left empty,
      all VM instances underneath this policy are targeted. At the same level
      in the resource hierarchy (that is within a project), the service
      prevents the creation of multiple policies that conflict with each
      other. For more information, see how the service [handles assignment
      conflicts](/compute/docs/os-config-management/create-guest-
      policy#handle-conflicts).
    createTime: Output only. Time this guest policy was created.
    description: Description of the guest policy. Length of the description is
      limited to 1024 characters.
    etag: The etag for this guest policy. If this is provided on update, it
      must match the server's etag.
    name: Required. Unique name of the resource in this project using one of
      the following forms:
      `projects/{project_number}/guestPolicies/{guest_policy_id}`.
    packageRepositories: A list of package repositories to configure on the VM
      instance. This is done before any other configs are applied so they can
      use these repos. Package repositories are only configured if the
      corresponding package manager(s) are available.
    packages: The software packages to be managed by this policy.
    recipes: A list of Recipes to install on the VM instance.
    updateTime: Output only. Last time this guest policy was updated.
  """
    assignment = _messages.MessageField('Assignment', 1)
    createTime = _messages.StringField(2)
    description = _messages.StringField(3)
    etag = _messages.StringField(4)
    name = _messages.StringField(5)
    packageRepositories = _messages.MessageField('PackageRepository', 6, repeated=True)
    packages = _messages.MessageField('Package', 7, repeated=True)
    recipes = _messages.MessageField('SoftwareRecipe', 8, repeated=True)
    updateTime = _messages.StringField(9)