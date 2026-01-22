from heat.common.i18n import _
from heat.engine import properties
from heat.engine.resources import alarm_base
from heat.engine import support
class AodhGnocchiAggregationByResourcesAlarm(AodhGnocchiResourcesAlarm):
    """A resource that implements alarm as an aggregation of resources alarms.

    A resource that implements alarm which uses aggregation of resources alarms
    with some condition. If state of a system is satisfied alarm condition,
    alarm is activated.
    """
    support_status = support.SupportStatus(version='2015.1')
    PROPERTIES = METRIC, QUERY, RESOURCE_TYPE = ('metric', 'query', 'resource_type')
    PROPERTIES += COMMON_GNOCCHI_PROPERTIES
    properties_schema = {METRIC: properties.Schema(properties.Schema.STRING, _('Metric name watched by the alarm.'), required=True, update_allowed=True), QUERY: properties.Schema(properties.Schema.STRING, _('The query to filter the metrics.'), required=True, update_allowed=True), RESOURCE_TYPE: properties.Schema(properties.Schema.STRING, _('Resource type.'), required=True, update_allowed=True)}
    properties_schema.update(common_gnocchi_properties_schema)
    properties_schema.update(alarm_base.common_properties_schema)
    alarm_type = 'gnocchi_aggregation_by_resources_threshold'