import copy
import random
from typing import List, Dict, Union
from parlai.core.agents import create_agents_from_shared
from parlai.core.loader import load_task_module, load_world_module
from parlai.core.metrics import aggregate_named_reports
from parlai.core.opt import Opt
from parlai.core.teachers import Teacher, create_task_agent_from_taskname
from parlai.utils.data import DatatypeHelper
from parlai.utils.misc import Timer, display_messages
from parlai.tasks.tasks import ids_to_tasks
import parlai.utils.logging as logging
class MultiAgentDialogWorld(World):
    """
    Basic world where each agent gets a turn in a round-robin fashion.

    Each agent receives as input the actions of all other agents since its last `act()`.
    """

    def __init__(self, opt: Opt, agents, shared=None):
        super().__init__(opt)
        if shared:
            self.agents = create_agents_from_shared(shared['agents'])
        else:
            self.agents = agents
        self.acts = [None] * len(self.agents)

    def parley(self):
        """
        Perform a turn for every agent.

        For each agent, get an observation of the last action each of the other agents
        took. Then take an action yourself.
        """
        acts = self.acts
        for index, agent in enumerate(self.agents):
            acts[index] = agent.act()
            for other_agent in self.agents:
                if other_agent != agent:
                    other_agent.observe(validate(acts[index]))
        self.update_counters()

    def get_task_agent(self):
        """
        Return task agent.
        """
        return self.get_agents()[0]

    def get_model_agent(self):
        """
        Return model agent.
        """
        return self.get_agents()[1]

    def epoch_done(self):
        """
        Return if the epoch is done for any subagent.
        """
        done = False
        for a in self.agents:
            if a.epoch_done():
                done = True
        return done

    def episode_done(self):
        """
        Return if the episode is done for any subagent.
        """
        done = False
        for a in self.agents:
            if a.episode_done():
                done = True
        return done

    def report(self):
        """
        Report metrics for all subagents.
        """
        metrics = {}
        for a in self.agents:
            if hasattr(a, 'report'):
                m = a.report()
                for k, v in m.items():
                    if k not in metrics:
                        metrics[k] = v
        if metrics and 'exs' in metrics:
            self.total_exs += metrics['exs'].value()
        return metrics

    def shutdown(self):
        """
        Shutdown each agent.
        """
        for a in self.agents:
            a.shutdown()