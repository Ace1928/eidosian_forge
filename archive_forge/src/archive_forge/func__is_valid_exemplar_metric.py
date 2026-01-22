from ..utils import floatToGoString
def _is_valid_exemplar_metric(metric, sample):
    if metric.type == 'counter' and sample.name.endswith('_total'):
        return True
    if metric.type in ('histogram', 'gaugehistogram') and sample.name.endswith('_bucket'):
        return True
    return False