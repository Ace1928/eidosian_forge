import datetime
import six
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.oauth2 import sts
class AccessBoundaryRule(object):
    """Defines an access boundary rule which contains information on the resource that
    the rule applies to, the upper bound of the permissions that are available on that
    resource and an optional condition to further restrict permissions.
    """

    def __init__(self, available_resource, available_permissions, availability_condition=None):
        """Instantiates a single access boundary rule.

        Args:
            available_resource (str): The full resource name of the Cloud Storage bucket
                that the rule applies to. Use the format
                "//storage.googleapis.com/projects/_/buckets/bucket-name".
            available_permissions (Sequence[str]): A list defining the upper bound that
                the downscoped token will have on the available permissions for the
                resource. Each value is the identifier for an IAM predefined role or
                custom role, with the prefix "inRole:". For example:
                "inRole:roles/storage.objectViewer".
                Only the permissions in these roles will be available.
            availability_condition (Optional[google.auth.downscoped.AvailabilityCondition]):
                Optional condition that restricts the availability of permissions to
                specific Cloud Storage objects.

        Raises:
            InvalidType: If any of the parameters are not of the expected types.
            InvalidValue: If any of the parameters are not of the expected values.
        """
        self.available_resource = available_resource
        self.available_permissions = available_permissions
        self.availability_condition = availability_condition

    @property
    def available_resource(self):
        """Returns the current available resource.

        Returns:
           str: The current available resource.
        """
        return self._available_resource

    @available_resource.setter
    def available_resource(self, value):
        """Updates the current available resource.

        Args:
            value (str): The updated value of the available resource.

        Raises:
            google.auth.exceptions.InvalidType: If the value is not a string.
        """
        if not isinstance(value, six.string_types):
            raise exceptions.InvalidType('The provided available_resource is not a string.')
        self._available_resource = value

    @property
    def available_permissions(self):
        """Returns the current available permissions.

        Returns:
           Tuple[str, ...]: The current available permissions. These are returned
               as an immutable tuple to prevent modification.
        """
        return tuple(self._available_permissions)

    @available_permissions.setter
    def available_permissions(self, value):
        """Updates the current available permissions.

        Args:
            value (Sequence[str]): The updated value of the available permissions.

        Raises:
            InvalidType: If the value is not a list of strings.
            InvalidValue: If the value is not valid.
        """
        for available_permission in value:
            if not isinstance(available_permission, six.string_types):
                raise exceptions.InvalidType('Provided available_permissions are not a list of strings.')
            if available_permission.find('inRole:') != 0:
                raise exceptions.InvalidValue("available_permissions must be prefixed with 'inRole:'.")
        self._available_permissions = list(value)

    @property
    def availability_condition(self):
        """Returns the current availability condition.

        Returns:
           Optional[google.auth.downscoped.AvailabilityCondition]: The current
               availability condition.
        """
        return self._availability_condition

    @availability_condition.setter
    def availability_condition(self, value):
        """Updates the current availability condition.

        Args:
            value (Optional[google.auth.downscoped.AvailabilityCondition]): The updated
                value of the availability condition.

        Raises:
            google.auth.exceptions.InvalidType: If the value is not of type google.auth.downscoped.AvailabilityCondition
                or None.
        """
        if not isinstance(value, AvailabilityCondition) and value is not None:
            raise exceptions.InvalidType("The provided availability_condition is not a 'google.auth.downscoped.AvailabilityCondition' or None.")
        self._availability_condition = value

    def to_json(self):
        """Generates the dictionary representation of the access boundary rule.
        This uses the format expected by the Security Token Service API as documented in
        `Defining a Credential Access Boundary`_.

        .. _Defining a Credential Access Boundary:
            https://cloud.google.com/iam/docs/downscoping-short-lived-credentials#define-boundary

        Returns:
            Mapping: The access boundary rule represented in a dictionary object.
        """
        json = {'availablePermissions': list(self.available_permissions), 'availableResource': self.available_resource}
        if self.availability_condition:
            json['availabilityCondition'] = self.availability_condition.to_json()
        return json