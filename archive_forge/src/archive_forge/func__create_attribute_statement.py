import datetime
import os
import subprocess  # nosec : see comments in the code below
import uuid
from oslo_log import log
from oslo_utils import fileutils
from oslo_utils import importutils
from oslo_utils import timeutils
import saml2
from saml2 import client_base
from saml2 import md
from saml2.profile import ecp
from saml2 import saml
from saml2 import samlp
from saml2.schema import soapenv
from saml2 import sigver
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
def _create_attribute_statement(self, user, user_domain_name, roles, project, project_domain_name, groups):
    """Create an object that represents a SAML AttributeStatement.

        <ns0:AttributeStatement>
            <ns0:Attribute Name="openstack_user">
                <ns0:AttributeValue
                  xsi:type="xs:string">test_user</ns0:AttributeValue>
            </ns0:Attribute>
            <ns0:Attribute Name="openstack_user_domain">
                <ns0:AttributeValue
                  xsi:type="xs:string">Default</ns0:AttributeValue>
            </ns0:Attribute>
            <ns0:Attribute Name="openstack_roles">
                <ns0:AttributeValue
                  xsi:type="xs:string">admin</ns0:AttributeValue>
                <ns0:AttributeValue
                  xsi:type="xs:string">member</ns0:AttributeValue>
            </ns0:Attribute>
            <ns0:Attribute Name="openstack_project">
                <ns0:AttributeValue
                  xsi:type="xs:string">development</ns0:AttributeValue>
            </ns0:Attribute>
            <ns0:Attribute Name="openstack_project_domain">
                <ns0:AttributeValue
                  xsi:type="xs:string">Default</ns0:AttributeValue>
            </ns0:Attribute>
            <ns0:Attribute Name="openstack_groups">
                <ns0:AttributeValue
                   xsi:type="xs:string">JSON:{"name":"group1","domain":{"name":"Default"}}
                </ns0:AttributeValue>
                <ns0:AttributeValue
                   xsi:type="xs:string">JSON:{"name":"group2","domain":{"name":"Default"}}
                </ns0:AttributeValue>
            </ns0:Attribute>

        </ns0:AttributeStatement>

        :returns: XML <AttributeStatement> object

        """

    def _build_attribute(attribute_name, attribute_values):
        attribute = saml.Attribute()
        attribute.name = attribute_name
        for value in attribute_values:
            attribute_value = saml.AttributeValue()
            attribute_value.set_text(value)
            attribute.attribute_value.append(attribute_value)
        return attribute
    user_attribute = _build_attribute('openstack_user', [user])
    roles_attribute = _build_attribute('openstack_roles', roles)
    project_attribute = _build_attribute('openstack_project', [project])
    project_domain_attribute = _build_attribute('openstack_project_domain', [project_domain_name])
    user_domain_attribute = _build_attribute('openstack_user_domain', [user_domain_name])
    attribute_statement = saml.AttributeStatement()
    attribute_statement.attribute.append(user_attribute)
    attribute_statement.attribute.append(roles_attribute)
    attribute_statement.attribute.append(project_attribute)
    attribute_statement.attribute.append(project_domain_attribute)
    attribute_statement.attribute.append(user_domain_attribute)
    if groups:
        groups_attribute = _build_attribute('openstack_groups', groups)
        attribute_statement.attribute.append(groups_attribute)
    return attribute_statement