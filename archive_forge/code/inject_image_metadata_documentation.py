from oslo_config import cfg
from taskflow.patterns import linear_flow as lf
from taskflow import task
from glance.i18n import _
Inject custom metadata properties to image

        :param image_id: Glance Image ID
        