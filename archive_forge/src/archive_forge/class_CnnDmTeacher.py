from parlai.core.teachers import MultiTaskTeacher
import parlai.tasks.squad.agents as squad
import parlai.tasks.iwslt14.agents as iwslt14
import parlai.tasks.cnn_dm.agents as cnn_dm
import parlai.tasks.multinli.agents as multinli
import parlai.tasks.sst.agents as sst
import parlai.tasks.qasrl.agents as qasrl
import parlai.tasks.qazre.agents as qazre
import parlai.tasks.woz.agents as woz
import parlai.tasks.wikisql.agents as wikisql
import parlai.tasks.mwsc.agents as mwsc
from copy import deepcopy
class CnnDmTeacher(cnn_dm.DefaultTeacher):
    pass