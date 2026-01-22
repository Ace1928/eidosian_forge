from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicePerimeter(_messages.Message):
    """`ServicePerimeter` describes a set of Google Cloud resources which can
  freely import and export data amongst themselves, but not export outside of
  the `ServicePerimeter`. If a request with a source within this
  `ServicePerimeter` has a target outside of the `ServicePerimeter`, the
  request is blocked. Otherwise the request is allowed. There are two types of
  service perimeter: regular and bridge. Regular perimeters cannot overlap, a
  single Google Cloud project or VPC network can only belong to a single
  regular perimeter. Perimeter bridges can contain only Google Cloud projects
  as members, a single Google Cloud project might belong to multiple Service
  perimeter bridges.

  Enums:
    PerimeterTypeValueValuesEnum: Perimeter type indicator. A single project
      or VPC network is allowed to be a member of single regular perimeter,
      but a project can be in multiple service perimeter bridges. A project
      cannot be a included in a perimeter bridge without being included in
      regular perimeter. For perimeter bridges, the restricted service list as
      well as access level lists must be empty.

  Fields:
    description: Description of the `ServicePerimeter` and its use. Does not
      affect behavior.
    name: Required. Resource name for the `ServicePerimeter`. Format:
      `accessPolicies/{access_policy}/servicePerimeters/{service_perimeter}`.
      The `service_perimeter` component must begin with a letter, followed by
      alphanumeric characters or `_`. After you create a `ServicePerimeter`,
      you cannot change its `name`.
    perimeterType: Perimeter type indicator. A single project or VPC network
      is allowed to be a member of single regular perimeter, but a project can
      be in multiple service perimeter bridges. A project cannot be a included
      in a perimeter bridge without being included in regular perimeter. For
      perimeter bridges, the restricted service list as well as access level
      lists must be empty.
    spec: Proposed (or dry run) ServicePerimeter configuration. This
      configuration allows to specify and test ServicePerimeter configuration
      without enforcing actual access restrictions. Only allowed to be set
      when the "use_explicit_dry_run_spec" flag is set.
    status: Current ServicePerimeter configuration. Specifies sets of
      resources, restricted services and access levels that determine
      perimeter content and boundaries.
    title: Human readable title. Must be unique within the Policy.
    useExplicitDryRunSpec: Use explicit dry run spec flag. Ordinarily, a dry-
      run spec implicitly exists for all Service Perimeters, and that spec is
      identical to the status for those Service Perimeters. When this flag is
      set, it inhibits the generation of the implicit spec, thereby allowing
      the user to explicitly provide a configuration ("spec") to use in a dry-
      run version of the Service Perimeter. This allows the user to test
      changes to the enforced config ("status") without actually enforcing
      them. This testing is done through analyzing the differences between
      currently enforced and suggested restrictions. use_explicit_dry_run_spec
      must bet set to True if any of the fields in the spec are set to non-
      default values.
  """

    class PerimeterTypeValueValuesEnum(_messages.Enum):
        """Perimeter type indicator. A single project or VPC network is allowed
    to be a member of single regular perimeter, but a project can be in
    multiple service perimeter bridges. A project cannot be a included in a
    perimeter bridge without being included in regular perimeter. For
    perimeter bridges, the restricted service list as well as access level
    lists must be empty.

    Values:
      PERIMETER_TYPE_REGULAR: Regular Perimeter. When no value is specified,
        the perimeter uses this type.
      PERIMETER_TYPE_BRIDGE: Perimeter Bridge.
    """
        PERIMETER_TYPE_REGULAR = 0
        PERIMETER_TYPE_BRIDGE = 1
    description = _messages.StringField(1)
    name = _messages.StringField(2)
    perimeterType = _messages.EnumField('PerimeterTypeValueValuesEnum', 3)
    spec = _messages.MessageField('ServicePerimeterConfig', 4)
    status = _messages.MessageField('ServicePerimeterConfig', 5)
    title = _messages.StringField(6)
    useExplicitDryRunSpec = _messages.BooleanField(7)