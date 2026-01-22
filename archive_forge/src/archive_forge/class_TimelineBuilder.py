class TimelineBuilder:
    """At each call to player.Player.update_texture we capture selected player
    state, before accepting the changes in the event. This is the same as
    capturing the state at the end of previous update call.
    Output is a sequence of tuples capturing the desired fields.
    Meant to extract info on behalf of other sw, especially visualization.
    """

    def __init__(self, recorded_events, events_definition=mp_events):
        mp = MediaPlayerStateIterator(recorded_events, events_definition, self.pre)
        self.mp_state_iterator = mp
        self.timeline = []

    def pre(self, event, st):
        if event[0] == 'p.P.ut.1.0':
            p = (st['wall_time'], st['pyglet_time'], st['audio_time'], st['current_time'], st['frame_num'], st['rescheduling_time'])
            self.timeline.append(p)

    def get_timeline(self):
        """remember video_time and audio_time can be None"""
        for st in self.mp_state_iterator:
            pass
        return self.timeline[1:]