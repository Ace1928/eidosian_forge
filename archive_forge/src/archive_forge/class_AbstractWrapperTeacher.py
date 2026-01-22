import copy
from abc import ABC, abstractmethod
from parlai.core.agents import create_agent_from_shared
from parlai.core.opt import Opt
from parlai.core.teachers import create_task_agent_from_taskname, Teacher
class AbstractWrapperTeacher(Teacher, ABC):
    """
    Abstract teacher that will wrap around another teacher and allow for manipulating
    the fields returned by the inner teacher.
    """

    @classmethod
    def add_cmdline_args(cls, parser):
        agent = parser.add_argument_group('AbstractWrapper args')
        agent.add_argument('-wt', '--wrapper-task', type=str, help='The task whose fields will be manipulated.')
        known_args, _ = parser.parse_known_args(nohelp=True)
        parser.add_task_args(known_args.wrapper_task)

    def __init__(self, opt: Opt, shared=None):
        if ',' in opt['task']:
            raise ValueError('AbstractWrapperTeacher cannot be used with multiple tasks!')
        self.id = opt['task']
        self.opt = opt
        if shared:
            self.task = create_agent_from_shared(shared['task'])
        else:
            opt_singletask = copy.deepcopy(opt)
            opt_singletask['task'] = opt['wrapper_task']
            self.task = create_task_agent_from_taskname(opt_singletask)[0]

    @abstractmethod
    def act(self):
        """
        Act on the previous observation.
        """
        raise NotImplementedError('Abstract class: user must implement act() method')

    def num_examples(self):
        """
        Return the number of examples.
        """
        return self.task.num_examples()

    def num_episodes(self):
        """
        Return the number of episodes.

        Because the dataset is flattened, there will be one episode per example.
        """
        return self.task.num_examples()

    def observe(self, observation):
        """
        Make an observation.
        """
        return self.task.observe(observation)

    def epoch_done(self):
        """
        Return whether the subtask is completed.
        """
        return self.task.epoch_done()

    def report(self):
        """
        Report metrics for the subtask.
        """
        return self.task.report()

    def reset(self):
        """
        Reset the subtask.
        """
        self.task.reset()

    def reset_metrics(self):
        """
        Reset metrics for the subtask.
        """
        self.task.reset_metrics()

    def save(self):
        """
        Save the subtask.
        """
        self.task.save()

    def share(self):
        """
        Share the subtask.
        """
        shared = {}
        shared['class'] = type(self)
        shared['opt'] = self.opt
        shared['task'] = self.task.share()
        return shared