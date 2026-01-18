from heat.common import exception
from heat.common.i18n import _
A class for specifying properties relationships.

    Properties group allows to specify relations between properties or other
    properties groups with operators AND, OR and XOR by one-key dict with list
    value. For example, if there are two properties: "subprop1", which is
    child of property "prop1", and property "prop2", and they should not be
    specified together, then properties group for them should be next::

      {XOR: [["prop1", "subprop1"], ["prop2"]]}

    where each property name should be set as list of strings. Also, if these
    properties are exclusive with properties "prop3" and "prop4", which should
    be specified both, then properties group will be defined such way::

      {XOR: [ ["prop1", "subprop1"], ["prop2"],
              {AND: [ ["prop3"], ["prop4"] ]} ]}

    where one-key dict with key "AND" is nested properties group.
    