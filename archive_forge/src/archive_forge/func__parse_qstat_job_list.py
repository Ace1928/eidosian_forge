import os
import pwd
import re
import subprocess
import time
import xml.dom.minidom
import random
from ... import logging
from ...interfaces.base import CommandLine
from .base import SGELikeBatchManagerBase, logger
def _parse_qstat_job_list(self, xml_job_list):
    current_jobs_parsed = list()
    for current_job_element in xml_job_list:
        try:
            job_queue_name = current_job_element.getElementsByTagName('queue_name')[0].childNodes[0].data
        except:
            job_queue_name = 'unknown'
        try:
            job_slots = int(current_job_element.getElementsByTagName('slots')[0].childNodes[0].data)
        except:
            job_slots = -1
        job_queue_state = current_job_element.getAttribute('state')
        job_num = int(current_job_element.getElementsByTagName('JB_job_number')[0].childNodes[0].data)
        try:
            job_time_text = current_job_element.getElementsByTagName('JAT_start_time')[0].childNodes[0].data
            job_time = float(time.mktime(time.strptime(job_time_text, '%Y-%m-%dT%H:%M:%S')))
        except:
            job_time = float(0.0)
        task_id = int(job_num)
        if task_id in self._task_dictionary:
            self._task_dictionary[task_id].update_info(job_queue_state, job_time, job_queue_name, job_slots)
            sge_debug_print('Updating job:  {0}'.format(self._task_dictionary[task_id]))
            current_jobs_parsed.append(task_id)
        else:
            self._out_of_scope_jobs.append(task_id)
    for dictionary_job in list(self._task_dictionary.keys()):
        if dictionary_job not in current_jobs_parsed:
            is_completed = self._qacct_verified_complete(dictionary_job)
            if is_completed:
                self._task_dictionary[dictionary_job].set_state('zombie')
            else:
                sge_debug_print('ERROR:  Job not in current parselist, and not in done list {0}: {1}'.format(dictionary_job, self._task_dictionary[dictionary_job]))
                pass
        if self._task_dictionary[dictionary_job].is_initializing():
            is_completed = self._qacct_verified_complete(dictionary_job)
            if is_completed:
                self._task_dictionary[dictionary_job].set_state('zombie')
            else:
                sge_debug_print('ERROR:  Job not in still in initialization mode, and not in done list {0}: {1}'.format(dictionary_job, self._task_dictionary[dictionary_job]))
                pass