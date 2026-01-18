import copy
from oslo_reports.models import base as base_model
from oslo_reports.views.json import generic as jsonviews
from oslo_reports.views.text import generic as textviews
from oslo_reports.views.xml import generic as xmlviews
A Model With Default Views of Various Types

    A model with default views has several predefined views,
    each associated with a given type.  This is often used for
    when a submodel should have an attached view, but the view
    differs depending on the serialization format

    Parameters are as the superclass, except for any
    parameters ending in '_view': these parameters
    get stored as default views.

    The default 'default views' are

    text
        :class:`oslo_reports.views.text.generic.KeyValueView`
    xml
        :class:`oslo_reports.views.xml.generic.KeyValueView`
    json
        :class:`oslo_reports.views.json.generic.KeyValueView`

    .. function:: to_type()

       ('type' is one of the 'default views' defined for this model)
       Serializes this model using the default view for 'type'

       :rtype: str
       :returns: this model serialized as 'type'
    