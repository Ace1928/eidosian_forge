import datetime
def _get_duration_components(duration):
    days = duration.days
    seconds = duration.seconds
    microseconds = duration.microseconds
    minutes = seconds // 60
    seconds %= 60
    hours = minutes // 60
    minutes %= 60
    return (days, hours, minutes, seconds, microseconds)