import os
from typing import Callable, Optional
import gym
from gym import logger
Save videos from rendering frames.

    This function extract video from a list of render frame episodes.

    Args:
        frames (List[RenderFrame]): A list of frames to compose the video.
        video_folder (str): The folder where the recordings will be stored
        episode_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this episode
        step_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this step
        video_length (int): The length of recorded episodes. If it isn't specified, the entire episode is recorded.
            Otherwise, snippets of the specified length are captured.
        name_prefix (str): Will be prepended to the filename of the recordings.
        episode_index (int): The index of the current episode.
        step_starting_index (int): The step index of the first frame.
        **kwargs: The kwargs that will be passed to moviepy's ImageSequenceClip.
            You need to specify either fps or duration.

    Example:
        >>> import gym
        >>> from gym.utils.save_video import save_video
        >>> env = gym.make("FrozenLake-v1", render_mode="rgb_array_list")
        >>> env.reset()
        >>> step_starting_index = 0
        >>> episode_index = 0
        >>> for step_index in range(199):
        ...    action = env.action_space.sample()
        ...    _, _, done, _ = env.step(action)
        ...    if done:
        ...       save_video(
        ...          env.render(),
        ...          "videos",
        ...          fps=env.metadata["render_fps"],
        ...          step_starting_index=step_starting_index,
        ...          episode_index=episode_index
        ...       )
        ...       step_starting_index = step_index + 1
        ...       episode_index += 1
        ...       env.reset()
        >>> env.close()
    