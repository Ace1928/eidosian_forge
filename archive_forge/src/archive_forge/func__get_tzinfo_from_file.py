def _get_tzinfo_from_file(tzfilename: str):
    with open(tzfilename, 'rb') as tzfile:
        if pytz:
            return pytz.tzfile.build_tzinfo('local', tzfile)
        else:
            return zoneinfo.ZoneInfo.from_file(tzfile)