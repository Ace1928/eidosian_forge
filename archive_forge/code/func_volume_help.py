import abc
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
def volume_help(release_track):
    """Generates the help text for all registered volume types."""
    hlp = []
    for _, volume_type in sorted(_supported_volume_types.items(), key=lambda t: t[0]):
        if release_track in volume_type.release_tracks():
            hlp.append(volume_type.generate_help())
    return '\n\n'.join(hlp)