from heat.db import api as db_api
from heat.engine import service
from heat.engine import stack
from heat.tests.convergence.framework import message_processor
from heat.tests.convergence.framework import message_queue
from heat.tests.convergence.framework import scenario_template
from heat.tests import utils
Converts the scenario template into hot template.