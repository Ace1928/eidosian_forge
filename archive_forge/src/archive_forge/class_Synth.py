from ctypes import *
from ctypes.util import find_library
import os
class Synth:
    """Synth represents a FluidSynth synthesizer"""

    def __init__(self, gain=0.2, samplerate=44100, channels=256, **kwargs):
        """Create new synthesizer object to control sound generation

        Optional keyword arguments:
        gain : scale factor for audio output, default is 0.2
        lower values are quieter, allow more simultaneous notes
        samplerate : output samplerate in Hz, default is 44100 Hz
        added capability for passing arbitrary fluid settings using args
        """
        self.settings = new_fluid_settings()
        self.setting('synth.gain', gain)
        self.setting('synth.sample-rate', float(samplerate))
        self.setting('synth.midi-channels', channels)
        for opt, val in kwargs.items():
            self.setting(opt, val)
        self.synth = new_fluid_synth(self.settings)
        self.audio_driver = None
        self.midi_driver = None
        self.router = None

    def setting(self, opt, val):
        """change an arbitrary synth setting, type-smart"""
        if isinstance(val, (str, bytes)):
            fluid_settings_setstr(self.settings, opt.encode(), val.encode())
        elif isinstance(val, int):
            fluid_settings_setint(self.settings, opt.encode(), val)
        elif isinstance(val, float):
            fluid_settings_setnum(self.settings, opt.encode(), c_double(val))

    def get_setting(self, opt):
        """get current value of an arbitrary synth setting"""
        val = c_int()
        if fluid_settings_getint(self.settings, opt.encode(), byref(val)) == FLUIDSETTING_EXISTS:
            return val.value
        strval = create_string_buffer(32)
        if fluid_settings_copystr(self.settings, opt.encode(), strval, 32) == FLUIDSETTING_EXISTS:
            return strval.value.decode()
        num = c_double()
        if fluid_settings_getnum(self.settings, opt.encode(), byref(num)) == FLUIDSETTING_EXISTS:
            return round(num.value, 6)
        return None

    def start(self, driver=None, device=None, midi_driver=None, midi_router=None):
        """Start audio output driver in separate background thread

        Call this function any time after creating the Synth object.
        If you don't call this function, use get_samples() to generate
        samples.

        Optional keyword argument:
        driver : which audio driver to use for output
        device : the device to use for audio output
        midi_driver : the midi driver to use for communicating with midi devices
        see http://www.fluidsynth.org/api/fluidsettings.xml for allowed values and defaults by platform
        """
        driver = driver or self.get_setting('audio.driver')
        device = device or self.get_setting('audio.%s.device' % driver)
        midi_driver = midi_driver or self.get_setting('midi.driver')
        self.setting('audio.driver', driver)
        self.setting('audio.%s.device' % driver, device)
        self.audio_driver = new_fluid_audio_driver(self.settings, self.synth)
        self.setting('midi.driver', midi_driver)
        self.router = new_fluid_midi_router(self.settings, fluid_synth_handle_midi_event, self.synth)
        if new_fluid_cmd_handler:
            new_fluid_cmd_handler(self.synth, self.router)
        else:
            fluid_synth_set_midi_router(self.synth, self.router)
        if midi_router == None:
            self.midi_driver = new_fluid_midi_driver(self.settings, fluid_midi_router_handle_midi_event, self.router)
            self.custom_router_callback = None
        else:
            self.custom_router_callback = CFUNCTYPE(c_int, c_void_p, c_void_p)(midi_router)
            self.midi_driver = new_fluid_midi_driver(self.settings, self.custom_router_callback, self.router)
        return FLUID_OK

    def delete(self):
        if self.audio_driver:
            delete_fluid_audio_driver(self.audio_driver)
        delete_fluid_synth(self.synth)
        delete_fluid_settings(self.settings)

    def sfload(self, filename, update_midi_preset=0):
        """Load SoundFont and return its ID"""
        return fluid_synth_sfload(self.synth, filename.encode(), update_midi_preset)

    def sfunload(self, sfid, update_midi_preset=0):
        """Unload a SoundFont and free memory it used"""
        return fluid_synth_sfunload(self.synth, sfid, update_midi_preset)

    def program_select(self, chan, sfid, bank, preset):
        """Select a program"""
        return fluid_synth_program_select(self.synth, chan, sfid, bank, preset)

    def program_unset(self, chan):
        """Set the preset of a MIDI channel to an unassigned state"""
        return fluid_synth_unset_program(self.synth, chan)

    def channel_info(self, chan):
        """get soundfont, bank, prog, preset name of channel"""
        if fluid_synth_get_channel_info is not None:
            info = fluid_synth_channel_info_t()
            fluid_synth_get_channel_info(self.synth, chan, byref(info))
            return (info.sfont_id, info.bank, info.program, info.name)
        else:
            sfontid, banknum, presetnum = self.program_info(chan)
            presetname = self.sfpreset_name(sfontid, banknum, presetnum)
            return (sfontid, banknum, presetnum, presetname)

    def program_info(self, chan):
        """get active soundfont, bank, prog on a channel"""
        if fluid_synth_get_program is not None:
            sfontid = c_int()
            banknum = c_int()
            presetnum = c_int()
            fluid_synth_get_program(self.synth, chan, byref(sfontid), byref(banknum), byref(presetnum))
            return (sfontid.value, banknum.value, presetnum.value)
        else:
            sfontid, banknum, prognum, presetname = self.channel_info(chan)
            return (sfontid, banknum, prognum)

    def sfpreset_name(self, sfid, bank, prenum):
        """Return name of a soundfont preset"""
        if fluid_synth_get_sfont_by_id is not None:
            sfont = fluid_synth_get_sfont_by_id(self.synth, sfid)
            preset = fluid_sfont_get_preset(sfont, bank, prenum)
            if not preset:
                return None
            return fluid_preset_get_name(preset).decode('ascii')
        else:
            sfontid, banknum, presetnum, presetname = self.channel_info(chan)
            return presetname

    def router_clear(self):
        if self.router is not None:
            fluid_midi_router_clear_rules(self.router)

    def router_default(self):
        if self.router is not None:
            fluid_midi_router_set_default_rules(self.router)

    def router_begin(self, type):
        """types are [note|cc|prog|pbend|cpress|kpress]"""
        if self.router is not None:
            if type == 'note':
                self.router.cmd_rule_type = 0
            elif type == 'cc':
                self.router.cmd_rule_type = 1
            elif type == 'prog':
                self.router.cmd_rule_type = 2
            elif type == 'pbend':
                self.router.cmd_rule_type = 3
            elif type == 'cpress':
                self.router.cmd_rule_type = 4
            elif type == 'kpress':
                self.router.cmd_rule_type = 5
            if 'self.router.cmd_rule' in globals():
                delete_fluid_midi_router_rule(self.router.cmd_rule)
            self.router.cmd_rule = new_fluid_midi_router_rule()

    def router_end(self):
        if self.router is not None:
            if self.router.cmd_rule is None:
                return
            if fluid_midi_router_add_rule(self.router, self.router.cmd_rule, self.router.cmd_rule_type) < 0:
                delete_fluid_midi_router_rule(self.router.cmd_rule)
            self.router.cmd_rule = None

    def router_chan(self, min, max, mul, add):
        if self.router is not None:
            fluid_midi_router_rule_set_chan(self.router.cmd_rule, min, max, mul, add)

    def router_par1(self, min, max, mul, add):
        if self.router is not None:
            fluid_midi_router_rule_set_param1(self.router.cmd_rule, min, max, mul, add)

    def router_par2(self, min, max, mul, add):
        if self.router is not None:
            fluid_midi_router_rule_set_param2(self.router.cmd_rule, min, max, mul, add)

    def set_reverb(self, roomsize=-1.0, damping=-1.0, width=-1.0, level=-1.0):
        """
        roomsize Reverb room size value (0.0-1.0)
        damping Reverb damping value (0.0-1.0)
        width Reverb width value (0.0-100.0)
        level Reverb level value (0.0-1.0)
        """
        if fluid_synth_set_reverb is not None:
            return fluid_synth_set_reverb(self.synth, roomsize, damping, width, level)
        else:
            set = 0
            if roomsize >= 0:
                set += 1
            if damping >= 0:
                set += 2
            if width >= 0:
                set += 4
            if level >= 0:
                set += 8
            return fluid_synth_set_reverb_full(self.synth, set, roomsize, damping, width, level)

    def set_chorus(self, nr=-1, level=-1.0, speed=-1.0, depth=-1.0, type=-1):
        """
        nr Chorus voice count (0-99, CPU time consumption proportional to this value)
        level Chorus level (0.0-10.0)
        speed Chorus speed in Hz (0.29-5.0)
        depth_ms Chorus depth (max value depends on synth sample rate, 0.0-21.0 is safe for sample rate values up to 96KHz)
        type Chorus waveform type (0=sine, 1=triangle)
        """
        if fluid_synth_set_chorus is not None:
            return fluid_synth_set_chorus(self.synth, nr, level, speed, depth, type)
        else:
            set = 0
            if nr >= 0:
                set += 1
            if level >= 0:
                set += 2
            if speed >= 0:
                set += 4
            if depth >= 0:
                set += 8
            if type >= 0:
                set += 16
            return fluid_synth_set_chorus_full(self.synth, set, nr, level, speed, depth, type)

    def set_reverb_roomsize(self, roomsize):
        if fluid_synth_set_reverb_roomsize is not None:
            return fluid_synth_set_reverb_roomsize(self.synth, roomsize)
        else:
            return self.set_reverb(roomsize=roomsize)

    def set_reverb_damp(self, damping):
        if fluid_synth_set_reverb_damp is not None:
            return fluid_synth_set_reverb_damp(self.synth, damping)
        else:
            return self.set_reverb(damping=damping)

    def set_reverb_level(self, level):
        if fluid_synth_set_reverb_level is not None:
            return fluid_synth_set_reverb_level(self.synth, level)
        else:
            return self.set_reverb(level=level)

    def set_reverb_width(self, width):
        if fluid_synth_set_reverb_width is not None:
            return fluid_synth_set_reverb_width(self.synth, width)
        else:
            return self.set_reverb(width=width)

    def set_chorus_nr(self, nr):
        if fluid_synth_set_chorus_nr is not None:
            return fluid_synth_set_chorus_nr(self.synth, nr)
        else:
            return self.set_chorus(nr=nr)

    def set_chorus_level(self, level):
        if fluid_synth_set_chorus_level is not None:
            return fluid_synth_set_chorus_level(self.synth, level)
        else:
            return self.set_chorus(leve=level)

    def set_chorus_speed(self, speed):
        if fluid_synth_set_chorus_speed is not None:
            return fluid_synth_set_chorus_speed(self.synth, speed)
        else:
            return self.set_chorus(speed=speed)

    def set_chorus_depth(self, depth):
        if fluid_synth_set_chorus_depth is not None:
            return fluid_synth_set_chorus_depth(self.synth, depth)
        else:
            return self.set_chorus(depth=depth)

    def set_chorus_type(self, type):
        if fluid_synth_set_chorus_type is not None:
            return fluid_synth_set_chorus_type(self.synth, type)
        else:
            return self.set_chorus(type=type)

    def get_reverb_roomsize(self):
        return fluid_synth_get_reverb_roomsize(self.synth)

    def get_reverb_damp(self):
        return fluid_synth_get_reverb_damp(self.synth)

    def get_reverb_level(self):
        return fluid_synth_get_reverb_level(self.synth)

    def get_reverb_width(self):
        return fluid_synth_get_reverb_width(self.synth)

    def get_chorus_nr(self):
        return fluid_synth_get_chorus_nr(self.synth)

    def get_chorus_level(self):
        return fluid_synth_get_reverb_level(self.synth)

    def get_chorus_speed(self):
        if fluid_synth_get_chorus_speed is not None:
            return fluid_synth_get_chorus_speed(self.synth)
        else:
            return fluid_synth_get_chorus_speed_Hz(self.synth)

    def get_chorus_depth(self):
        if fluid_synth_get_chorus_depth is not None:
            return fluid_synth_get_chorus_depth(self.synth)
        else:
            return fluid_synth_get_chorus_depth_ms(self.synth)

    def get_chorus_type(self):
        return fluid_synth_get_chorus_type(self.synth)

    def noteon(self, chan, key, vel):
        """Play a note"""
        if key < 0 or key > 127:
            return False
        if chan < 0:
            return False
        if vel < 0 or vel > 127:
            return False
        return fluid_synth_noteon(self.synth, chan, key, vel)

    def noteoff(self, chan, key):
        """Stop a note"""
        if key < 0 or key > 127:
            return False
        if chan < 0:
            return False
        return fluid_synth_noteoff(self.synth, chan, key)

    def pitch_bend(self, chan, val):
        """Adjust pitch of a playing channel by small amounts

        A pitch bend value of 0 is no pitch change from default.
        A value of -2048 is 1 semitone down.
        A value of 2048 is 1 semitone up.
        Maximum values are -8192 to +8192 (transposing by 4 semitones).

        """
        return fluid_synth_pitch_bend(self.synth, chan, val + 8192)

    def cc(self, chan, ctrl, val):
        """Send control change value

        The controls that are recognized are dependent on the
        SoundFont.  Values are always 0 to 127.  Typical controls
        include:
          1 : vibrato
          7 : volume
          10 : pan (left to right)
          11 : expression (soft to loud)
          64 : sustain
          91 : reverb
          93 : chorus
        """
        return fluid_synth_cc(self.synth, chan, ctrl, val)

    def get_cc(self, chan, num):
        i = c_int()
        fluid_synth_get_cc(self.synth, chan, num, byref(i))
        return i.value

    def program_change(self, chan, prg):
        """Change the program"""
        return fluid_synth_program_change(self.synth, chan, prg)

    def bank_select(self, chan, bank):
        """Choose a bank"""
        return fluid_synth_bank_select(self.synth, chan, bank)

    def all_notes_off(self, chan):
        """Turn off all notes on a channel (release all keys)"""
        return fluid_synth_all_notes_off(self.synth, chan)

    def all_sounds_off(self, chan):
        """Turn off all sounds on a channel (equivalent to mute)"""
        return fluid_synth_all_sounds_off(self.synth, chan)

    def sfont_select(self, chan, sfid):
        """Choose a SoundFont"""
        return fluid_synth_sfont_select(self.synth, chan, sfid)

    def program_reset(self):
        """Reset the programs on all channels"""
        return fluid_synth_program_reset(self.synth)

    def system_reset(self):
        """Stop all notes and reset all programs"""
        return fluid_synth_system_reset(self.synth)

    def get_samples(self, len=1024):
        """Generate audio samples

        The return value will be a NumPy array containing the given
        length of audio samples.  If the synth is set to stereo output
        (the default) the array will be size 2 * len.

        """
        return fluid_synth_write_s16_stereo(self.synth, len)

    def tuning_dump(self, bank, prog, pitch):
        return fluid_synth_tuning_dump(self.synth, bank, prog, name.encode(), length(name), pitch)

    def midi_event_get_type(self, event):
        return fluid_midi_event_get_type(event)

    def midi_event_get_velocity(self, event):
        return fluid_midi_event_get_velocity(event)

    def midi_event_get_key(self, event):
        return fluid_midi_event_get_key(event)

    def midi_event_get_channel(self, event):
        return fluid_midi_event_get_channel(event)

    def midi_event_get_control(self, event):
        return fluid_midi_event_get_control(event)

    def midi_event_get_program(self, event):
        return fluid_midi_event_get_program(event)

    def midi_event_get_value(self, event):
        return fluid_midi_event_get_value(event)

    def play_midi_file(self, filename):
        self.player = new_fluid_player(self.synth)
        if self.player == None:
            return FLUID_FAILED
        if self.custom_router_callback != None:
            fluid_player_set_playback_callback(self.player, self.custom_router_callback, self.synth)
        status = fluid_player_add(self.player, filename.encode())
        if status == FLUID_FAILED:
            return status
        status = fluid_player_play(self.player)
        return status

    def play_midi_stop(self):
        status = fluid_player_stop(self.player)
        if status == FLUID_FAILED:
            return status
        status = fluid_player_seek(self.player, 0)
        delete_fluid_player(self.player)
        return status

    def player_set_tempo(self, tempo_type, tempo):
        return fluid_player_set_tempo(self.player, tempo_type, tempo)