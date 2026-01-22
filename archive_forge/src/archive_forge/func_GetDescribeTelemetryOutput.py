from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.apphub import consts
def GetDescribeTelemetryOutput(get_telemetry_response):
    return {consts.Telemetry.NAME: get_telemetry_response.name, consts.Telemetry.PROJECT: get_telemetry_response.project, consts.Telemetry.MONITORING_ENABLED: bool(get_telemetry_response.monitoringEnabled)}