import datetime
import six
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.oauth2 import sts
class AvailabilityCondition(object):
    """An optional condition that can be used as part of a Credential Access Boundary
    to further restrict permissions."""

    def __init__(self, expression, title=None, description=None):
        """Instantiates an availability condition using the provided expression and
        optional title or description.

        Args:
            expression (str): A condition expression that specifies the Cloud Storage
                objects where permissions are available. For example, this expression
                makes permissions available for objects whose name starts with "customer-a":
                "resource.name.startsWith('projects/_/buckets/example-bucket/objects/customer-a')"
            title (Optional[str]): An optional short string that identifies the purpose of
                the condition.
            description (Optional[str]): Optional details about the purpose of the condition.

        Raises:
            InvalidType: If any of the parameters are not of the expected types.
            InvalidValue: If any of the parameters are not of the expected values.
        """
        self.expression = expression
        self.title = title
        self.description = description

    @property
    def expression(self):
        """Returns the current condition expression.

        Returns:
           str: The current conditon expression.
        """
        return self._expression

    @expression.setter
    def expression(self, value):
        """Updates the current condition expression.

        Args:
            value (str): The updated value of the condition expression.

        Raises:
            google.auth.exceptions.InvalidType: If the value is not of type string.
        """
        if not isinstance(value, six.string_types):
            raise exceptions.InvalidType('The provided expression is not a string.')
        self._expression = value

    @property
    def title(self):
        """Returns the current title.

        Returns:
           Optional[str]: The current title.
        """
        return self._title

    @title.setter
    def title(self, value):
        """Updates the current title.

        Args:
            value (Optional[str]): The updated value of the title.

        Raises:
            google.auth.exceptions.InvalidType: If the value is not of type string or None.
        """
        if not isinstance(value, six.string_types) and value is not None:
            raise exceptions.InvalidType('The provided title is not a string or None.')
        self._title = value

    @property
    def description(self):
        """Returns the current description.

        Returns:
           Optional[str]: The current description.
        """
        return self._description

    @description.setter
    def description(self, value):
        """Updates the current description.

        Args:
            value (Optional[str]): The updated value of the description.

        Raises:
            google.auth.exceptions.InvalidType: If the value is not of type string or None.
        """
        if not isinstance(value, six.string_types) and value is not None:
            raise exceptions.InvalidType('The provided description is not a string or None.')
        self._description = value

    def to_json(self):
        """Generates the dictionary representation of the availability condition.
        This uses the format expected by the Security Token Service API as documented in
        `Defining a Credential Access Boundary`_.

        .. _Defining a Credential Access Boundary:
            https://cloud.google.com/iam/docs/downscoping-short-lived-credentials#define-boundary

        Returns:
            Mapping[str, str]: The availability condition represented in a dictionary
                object.
        """
        json = {'expression': self.expression}
        if self.title:
            json['title'] = self.title
        if self.description:
            json['description'] = self.description
        return json