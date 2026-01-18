import os
import json
import numpy as np
import ipywidgets as widgets
import pythreejs
import ipywebrtc
from IPython.display import display
def update_keyframes(self):
    with self.output:
        options = [(self.format_keyframe(t, p, q), i) for i, (t, p, q) in enumerate(zip(self.times, self.positions, self.quaternions))]
        self.select_keyframes.options = options
        self.position_track = pythreejs.VectorKeyframeTrack(name='.position', times=self.times, values=self.positions, interpolation=self.select_interpolation.value)
        self.rotation_track = pythreejs.QuaternionKeyframeTrack(name='.quaternion', times=self.times, values=self.quaternions, interpolation=self.select_interpolation.value)
        if len(self.positions):
            self.camera_clip = pythreejs.AnimationClip(tracks=[self.position_track, self.rotation_track])
            self.mixer = pythreejs.AnimationMixer(self.camera)
            self.camera_action = pythreejs.AnimationAction(self.mixer, self.camera_clip, self.camera)
            self.camera_action_box.children = [self.camera_action]
        else:
            self.camera_action_box.children = []