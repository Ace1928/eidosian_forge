import re
from prometheus_client import start_http_server
from prometheus_client.core import (
from opencensus.common.transports import sync
from opencensus.stats import aggregation_data as aggregation_data_module
from opencensus.stats import base_exporter
import logging
def to_metric(self, desc, tag_values, agg_data, metrics_map):
    """to_metric translate the data that OpenCensus create
        to Prometheus format, using Prometheus Metric object
        :type desc: dict
        :param desc: The map that describes view definition
        :type tag_values: tuple of :class:
            `~opencensus.tags.tag_value.TagValue`
        :param object of opencensus.tags.tag_value.TagValue:
            TagValue object used as label values
        :type agg_data: object of :class:
            `~opencensus.stats.aggregation_data.AggregationData`
        :param object of opencensus.stats.aggregation_data.AggregationData:
            Aggregated data that needs to be converted as Prometheus samples
        :rtype: :class:`~prometheus_client.core.CounterMetricFamily` or
                :class:`~prometheus_client.core.HistogramMetricFamily` or
                :class:`~prometheus_client.core.UnknownMetricFamily` or
                :class:`~prometheus_client.core.GaugeMetricFamily`
        :returns: A Prometheus metric object
        """
    metric_name = desc['name']
    metric_description = desc['documentation']
    label_keys = desc['labels']
    metric_units = desc['units']
    assert len(tag_values) == len(label_keys), (tag_values, label_keys)
    tag_values = [tv if tv else '' for tv in tag_values]
    if isinstance(agg_data, aggregation_data_module.CountAggregationData):
        metric = metrics_map.get(metric_name)
        if not metric:
            metric = CounterMetricFamily(name=metric_name, documentation=metric_description, unit=metric_units, labels=label_keys)
            metrics_map[metric_name] = metric
        metric.add_metric(labels=tag_values, value=agg_data.count_data)
        return metric
    elif isinstance(agg_data, aggregation_data_module.DistributionAggregationData):
        assert agg_data.bounds == sorted(agg_data.bounds)
        buckets = []
        cum_count = 0
        for ii, bound in enumerate(agg_data.bounds):
            cum_count += agg_data.counts_per_bucket[ii]
            bucket = [str(bound), cum_count]
            buckets.append(bucket)
        buckets.append(['+Inf', agg_data.count_data])
        metric = metrics_map.get(metric_name)
        if not metric:
            metric = HistogramMetricFamily(name=metric_name, documentation=metric_description, labels=label_keys)
            metrics_map[metric_name] = metric
        metric.add_metric(labels=tag_values, buckets=buckets, sum_value=agg_data.sum)
        return metric
    elif isinstance(agg_data, aggregation_data_module.SumAggregationData):
        metric = metrics_map.get(metric_name)
        if not metric:
            metric = UnknownMetricFamily(name=metric_name, documentation=metric_description, labels=label_keys)
            metrics_map[metric_name] = metric
        metric.add_metric(labels=tag_values, value=agg_data.sum_data)
        return metric
    elif isinstance(agg_data, aggregation_data_module.LastValueAggregationData):
        metric = metrics_map.get(metric_name)
        if not metric:
            metric = GaugeMetricFamily(name=metric_name, documentation=metric_description, labels=label_keys)
            metrics_map[metric_name] = metric
        metric.add_metric(labels=tag_values, value=agg_data.value)
        return metric
    else:
        raise ValueError(f'unsupported aggregation type {type(agg_data)}')