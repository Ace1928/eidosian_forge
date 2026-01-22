import gym
import json
import numpy as np
from copy import deepcopy
from minerl.herobraine.hero import mc, handlers
from collections import defaultdict, deque
class MinecraftReplayWrapper(ReplayWrapper):
    """
    Minecraft-specific implementation of the ReplayWrapper.
    Staying on the recorded trajectory is judged by difference between
    current vs recorded agent coordinates as well as by differnce between
    current and recorded agent inventories.

    :param replay_file:       see ReplayWrapper
    :param clip_stats:        if True, reported stats are adjusted by the amount at the end of the replay
                              so that various monitors only count stats achieved by the policy, not
                              during the replay.
    :param max_steps:         do not replay for more than this number of steps
    :param gui_camera_scaler: additional factor to multiply replay camera actions when gui is open.
                              Useful when replaying data recorded with older (<=5.8) versions of 
                              minerec recorder (should be set to 0.5)
    """

    def __init__(self, env, replay_file, clip_stats=True, max_steps=None, gui_camera_scaler=1.0, replay_on_reset=False):
        super().__init__(env, replay_file, max_steps=max_steps, replay_on_reset=replay_on_reset)
        self.last_info = None
        self.last_ob = None
        self.clip_stats = clip_stats
        self.multiagent = False
        self.gui_camera_scaler = gui_camera_scaler
        self.mismatched_ticks = 0
        self.max_mismatched_ticks = 20
        self.max_dcoord = 3
        self._patch_agent_start()

    def is_on_trajectory(self, replay_action):
        """
        Checks if the environment is still following the recorded trajectory
        by comparing recorded and current coordinates, and inventories.
        :param replay_action: current action to be replayed. Assumed to be a dict,
                              with xpos, ypos, zpos, and inventory, that are utilized
                              to compare agent location and inventory to the one reported by env
                            
        """
        if self.last_info is None or self.last_ob is None:
            return True
        if self.multiagent:
            return self.is_on_trajectory_impl(replay_action, self.last_ob['agent_0'], self.last_info['agent_0'])
        else:
            return self.is_on_trajectory_impl(replay_action, self.last_ob, self.last_info)

    def is_on_trajectory_impl(self, replay_action, ob, info):
        max_dcoord = self.max_dcoord
        location_stats = info['location_stats']
        x = location_stats['xpos']
        y = location_stats['ypos']
        z = location_stats['zpos']
        yaw = location_stats['yaw']
        pitch = location_stats['pitch']
        x1 = replay_action['xpos']
        y1 = replay_action['ypos']
        z1 = replay_action['zpos']
        tick1 = replay_action['tick']
        yaw1 = replay_action['yaw']
        pitch1 = replay_action['pitch']
        if abs(x - x1) > max_dcoord or abs(y - y1) > max_dcoord or abs(z - z1) > max_dcoord or (abs(yaw - yaw1) > max_dcoord) or (abs(pitch - pitch1) > max_dcoord):
            print(f'Tick {tick1}: Coords mismatch: is {x}, {y}, {z}, {yaw}, {pitch}, should be {x1}, {y1}, {z1}, {yaw1}, {pitch1}')
            self.mismatched_ticks += 1
        elif 'inventory' in replay_action and (not inventory_matches(ob['inventory'], replay_action['inventory'])):
            print(f'Tick {tick1}: Inventory mismatch')
            self.mismatched_ticks += 1
        else:
            self.mismatched_ticks = 0
        return self.mismatched_ticks < self.max_mismatched_ticks

    def replay2env(self, replay_action, next_action):
        self.last_action = replay_action
        ac = mc.minerec_to_minerl_action(replay_action, next_action=next_action, gui_camera_scaler=self.gui_camera_scaler, esc_to_inventory=False)
        if self.multiagent:
            ac = {'agent_0': ac}
        return ac

    def step(self, ac):
        ob, rew, done, info = super().step(ac)
        self.update_stats(ob, info)
        if self.clip_stats:
            ob = self._clip_stats(ob)
        return (ob, rew, done, info)

    def reset(self):
        ob = super().reset()
        self.multiagent = 'agent_0' in ob
        self.mismatched_ticks = 0
        self.last_ob = ob
        self.last_info = None
        return ob

    def update_stats(self, ob, info):
        replaying = info[ReplayWrapper.IGNORE_POLICY_ACTION]
        if replaying:
            self.last_info = deepcopy(info)
            self.last_ob = deepcopy(ob)
        if self.multiagent:
            info['agent_0'][ReplayWrapper.IGNORE_POLICY_ACTION] = replaying

    def _clip_stats(self, ob):
        """
        Adjusts stats (currently, only inventory) by the amount at the end of the replay
        """
        if self.multiagent:
            return {'agent_0': subtract_stats(ob['agent_0'], self.last_ob['agent_0'])}
        else:
            return subtract_stats(ob, self.last_ob)

    def _patch_agent_start(self):
        old_create_agent_start = self.task.create_agent_start

        def create_agent_start():
            h = old_create_agent_start()
            start_pos = self._get_start_pos()
            start_velocity = self._get_start_velocity()
            if start_pos is not None:
                h.append(handlers.AgentStartPlacement(*start_pos))
            if start_velocity is not None:
                h.append(handlers.AgentStartVelocity(*start_velocity))
            return h
        self.task.create_agent_start = create_agent_start

    def _get_start_pos(self):
        if len(self.actions) == 0:
            return None
        a = self.actions[0]
        return (a['xpos'], a['ypos'], a['zpos'], a['yaw'], a['pitch'])

    def _get_start_velocity(self):
        if len(self.actions) < 2:
            return None
        a, a1 = (self.actions[0], self.actions[1])
        vx = a1['xpos'] - a['xpos']
        vy = a1['ypos'] - a['ypos']
        vz = a1['zpos'] - a['zpos']
        return (vx, vy, vz)

    def extra_steps_on_reset(self, ob):
        for i in range(len(self.actions) - 1):
            a, na = (self.actions[i], self.actions[i + 1])
            sprint_stat = 'minecraft.custom:minecraft.sprint_one_cm'
            if na.get('stats', {}).get(sprint_stat, 0) <= a.get('stats', {}).get(sprint_stat, 0):
                break
            a['keyboard']['keys'].append('key.keyboard.left.control')
        replay_action = self.actions[0]
        if replay_action.get('isGuiOpen', False):
            self.env.step(self.env.action_space.no_op())
            ac = self.env.action_space.no_op()
            ac['inventory' if replay_action.get('isGuiInventory') else 'use'] = 1
            self.env.step(ac)
            for _ in range(5):
                self.env.step(self.env.action_space.no_op())
            ma = replay_action['mouse']
            dx = (ma['x'] - 640) / 2
            dy = (ma['y'] - 360) / 2
            dx = ma.get('scaledX', dx)
            dy = ma.get('scaledY', dy)
            ac = self.env.action_space.no_op()
            ac['camera'] = mc.mouse_to_camera({'dx': dx, 'dy': dy})
            ob, _, _, _ = self.env.step(ac)
        return ob