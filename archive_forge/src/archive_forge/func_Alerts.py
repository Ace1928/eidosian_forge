import threading
from tensorboard import errors
def Alerts(self, run, begin, end, alert_type_filter=None):
    """Get alerts from the debugged TensorFlow program.

        Args:
          run: The tfdbg2 run to get Alerts from.
          begin: Beginning alert index.
          end: Ending alert index.
          alert_type_filter: Optional filter string for alert type, used to
            restrict retrieved alerts data to a single type. If used,
            `begin` and `end` refer to the beginning and ending indices within
            the filtered alert type.
        """
    from tensorflow.python.debug.lib import debug_events_monitors
    runs = self.Runs()
    if run not in runs:
        return None
    alerts = []
    alerts_breakdown = dict()
    alerts_by_type = dict()
    for monitor in self._monitors:
        monitor_alerts = monitor.alerts()
        if not monitor_alerts:
            continue
        alerts.extend(monitor_alerts)
        if isinstance(monitor, debug_events_monitors.InfNanMonitor):
            alert_type = 'InfNanAlert'
        else:
            alert_type = '__MiscellaneousAlert__'
        alerts_breakdown[alert_type] = len(monitor_alerts)
        alerts_by_type[alert_type] = monitor_alerts
    num_alerts = len(alerts)
    if alert_type_filter is not None:
        if alert_type_filter not in alerts_breakdown:
            raise errors.InvalidArgumentError('Filtering of alerts failed: alert type %s does not exist' % alert_type_filter)
        alerts = alerts_by_type[alert_type_filter]
    end = self._checkBeginEndIndices(begin, end, len(alerts))
    return {'begin': begin, 'end': end, 'alert_type': alert_type_filter, 'num_alerts': num_alerts, 'alerts_breakdown': alerts_breakdown, 'per_type_alert_limit': DEFAULT_PER_TYPE_ALERT_LIMIT, 'alerts': [_alert_to_json(alert) for alert in alerts[begin:end]]}