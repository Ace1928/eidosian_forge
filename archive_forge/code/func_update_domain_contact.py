import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.route53.domains import exceptions
def update_domain_contact(self, domain_name, admin_contact=None, registrant_contact=None, tech_contact=None):
    """
        This operation updates the contact information for a
        particular domain. Information for at least one contact
        (registrant, administrator, or technical) must be supplied for
        update.

        If the update is successful, this method returns an operation
        ID that you can use to track the progress and completion of
        the action. If the request is not completed successfully, the
        domain registrant will be notified by email.

        :type domain_name: string
        :param domain_name: The name of a domain.
        Type: String

        Default: None

        Constraints: The domain name can contain only the letters a through z,
            the numbers 0 through 9, and hyphen (-). Internationalized Domain
            Names are not supported.

        Required: Yes

        :type admin_contact: dict
        :param admin_contact: Provides detailed contact information.
        Type: Complex

        Children: `FirstName`, `MiddleName`, `LastName`, `ContactType`,
            `OrganizationName`, `AddressLine1`, `AddressLine2`, `City`,
            `State`, `CountryCode`, `ZipCode`, `PhoneNumber`, `Email`, `Fax`,
            `ExtraParams`

        Required: Yes

        :type registrant_contact: dict
        :param registrant_contact: Provides detailed contact information.
        Type: Complex

        Children: `FirstName`, `MiddleName`, `LastName`, `ContactType`,
            `OrganizationName`, `AddressLine1`, `AddressLine2`, `City`,
            `State`, `CountryCode`, `ZipCode`, `PhoneNumber`, `Email`, `Fax`,
            `ExtraParams`

        Required: Yes

        :type tech_contact: dict
        :param tech_contact: Provides detailed contact information.
        Type: Complex

        Children: `FirstName`, `MiddleName`, `LastName`, `ContactType`,
            `OrganizationName`, `AddressLine1`, `AddressLine2`, `City`,
            `State`, `CountryCode`, `ZipCode`, `PhoneNumber`, `Email`, `Fax`,
            `ExtraParams`

        Required: Yes

        """
    params = {'DomainName': domain_name}
    if admin_contact is not None:
        params['AdminContact'] = admin_contact
    if registrant_contact is not None:
        params['RegistrantContact'] = registrant_contact
    if tech_contact is not None:
        params['TechContact'] = tech_contact
    return self.make_request(action='UpdateDomainContact', body=json.dumps(params))