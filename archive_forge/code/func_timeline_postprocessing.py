def timeline_postprocessing(timeline):
    """ Eliminates Nones in timeline so other software don't error.
        Extra lists are built for the vars with nones, each list with one point
        for each None in the form (wall_time, prev_value).
    """
    current_time_nones = []
    audio_time_nones = []
    old_current_time = 0
    old_audio_time = 0
    filtered_timeline = []
    for wall_time, pt, audio_time, current_time, fnum, rt in timeline:
        if current_time is None:
            current_time = old_current_time
            current_time_nones.append((wall_time, old_current_time))
        else:
            current_time_time = current_time
        if audio_time is None:
            audio_time = old_audio_time
            audio_time_nones.append((wall_time, old_audio_time))
        else:
            old_audio_time = audio_time
        filtered_timeline.append((wall_time, pt, audio_time, current_time, fnum, rt))
    return (filtered_timeline, current_time_nones, audio_time_nones)