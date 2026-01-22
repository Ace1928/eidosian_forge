import io
import os
import re
import sys
import time
import socket
import base64
import tempfile
import logging
from pyomo.common.dependencies import attempt_import
class kestrelAMPL(object):

    def __init__(self):
        self.setup_connection()

    def __del__(self):
        if self.neos is not None:
            self.transport.close()

    def setup_connection(self):
        import http.client
        proxy = os.environ.get('http_proxy', os.environ.get('HTTP_PROXY', ''))
        if NEOS.scheme == 'https':
            proxy = os.environ.get('https_proxy', os.environ.get('HTTPS_PROXY', proxy))
        if proxy:
            self.transport = ProxiedTransport()
            self.transport.set_proxy(proxy)
        elif NEOS.scheme == 'https':
            self.transport = xmlrpclib.SafeTransport()
        else:
            self.transport = xmlrpclib.Transport()
        self.neos = xmlrpclib.ServerProxy('%s://%s:%s' % (NEOS.scheme, NEOS.host, NEOS.port), transport=self.transport)
        logger.info('Connecting to the NEOS server ... ')
        try:
            result = self.neos.ping()
            logger.info('OK.')
        except (socket.error, xmlrpclib.ProtocolError, http.client.BadStatusLine):
            e = sys.exc_info()[1]
            self.neos = None
            logger.info('Fail: %s' % (e,))
            logger.warning('NEOS is temporarily unavailable:\n\t(%s)' % (e,))

    def tempfile(self):
        return os.path.join(tempfile.gettempdir(), 'at%s.jobs' % os.getenv('ampl_id'))

    def kill(self, jobNumber, password):
        response = self.neos.killJob(jobNumber, password)
        logger.info(response)

    def solvers(self):
        if self.neos is None:
            return []
        else:
            attempt = 0
            while attempt < 3:
                try:
                    return self.neos.listSolversInCategory('kestrel')
                except socket.timeout:
                    attempt += 1
            return []

    def retrieve(self, stub, jobNumber, password):
        results = self.neos.getFinalResults(jobNumber, password)
        if isinstance(results, xmlrpclib.Binary):
            results = results.data
        if stub[-4:] == '.sol':
            stub = stub[:-4]
        solfile = open(stub + '.sol', 'wb')
        solfile.write(results)
        solfile.close()

    def submit(self, xml):
        user = self.getEmailAddress()
        jobNumber, password = self.neos.submitJob(xml, user, 'kestrel')
        if jobNumber == 0:
            raise RuntimeError('%s\n\tJob not submitted' % (password,))
        logger.info("Job %d submitted to NEOS, password='%s'\n" % (jobNumber, password))
        logger.info('Check the following URL for progress report :\n')
        logger.info('%s://www.neos-server.org/neos/cgi-bin/nph-neos-solver.cgi?admin=results&jobnumber=%d&pass=%s\n' % (NEOS.scheme, jobNumber, password))
        return (jobNumber, password)

    def getEmailAddress(self):
        email = os.environ.get('NEOS_EMAIL', '')
        if _email_re.match(email):
            return email
        raise RuntimeError("NEOS requires a valid email address. Please set the 'NEOS_EMAIL' environment variable.")

    def getJobAndPassword(self):
        """
        If kestrel_options is set to job/password, then return
        the job and password values
        """
        jobNumber = 0
        password = ''
        options = os.getenv('kestrel_options')
        if options is not None:
            m = re.search('job\\s*=\\s*(\\d+)', options, re.IGNORECASE)
            if m:
                jobNumber = int(m.groups()[0])
            m = re.search('password\\s*=\\s*(\\S+)', options, re.IGNORECASE)
            if m:
                password = m.groups()[0]
        return (jobNumber, password)

    def getSolverName(self):
        """
        Read in the kestrel_options to pick out the solver name.
        The tricky parts:
          we don't want to be case sensitive, but NEOS is.
          we need to read in options variable
        """
        allKestrelSolvers = self.neos.listSolversInCategory('kestrel')
        kestrelAmplSolvers = []
        for s in allKestrelSolvers:
            i = s.find(':AMPL')
            if i > 0:
                kestrelAmplSolvers.append(s[0:i])
        self.options = None
        if 'kestrel_options' in os.environ:
            self.options = os.getenv('kestrel_options')
        elif 'KESTREL_OPTIONS' in os.environ:
            self.options = os.getenv('KESTREL_OPTIONS')
        if self.options is not None:
            m = re.search('solver\\s*=*\\s*(\\S+)', self.options, re.IGNORECASE)
            NEOS_solver_name = None
            if m:
                solver_name = m.groups()[0]
                for s in kestrelAmplSolvers:
                    if s.upper() == solver_name.upper():
                        NEOS_solver_name = s
                        break
                if not NEOS_solver_name:
                    raise RuntimeError('%s is not available on NEOS.  Choose from:\n\t%s' % (solver_name, '\n\t'.join(kestrelAmplSolvers)))
        if self.options is None or m is None:
            raise RuntimeError('%s is not available on NEOS.  Choose from:\n\t%s' % (solver_name, '\n\t'.join(kestrelAmplSolvers)))
        return NEOS_solver_name

    def formXML(self, stub):
        solver = self.getSolverName()
        zipped_nl_file = io.BytesIO()
        if os.path.exists(stub) and stub[-3:] == '.nl':
            stub = stub[:-3]
        nlfile = open(stub + '.nl', 'rb')
        zipper = gzip.GzipFile(mode='wb', fileobj=zipped_nl_file)
        zipper.write(nlfile.read())
        zipper.close()
        nlfile.close()
        ampl_files = {}
        for key in ['adj', 'col', 'env', 'fix', 'spc', 'row', 'slc', 'unv']:
            if os.access(stub + '.' + key, os.R_OK):
                f = open(stub + '.' + key, 'r')
                val = ''
                buf = f.read()
                while buf:
                    val += buf
                    buf = f.read()
                f.close()
                ampl_files[key] = val
        priority = ''
        m = re.search('priority[\\s=]+(\\S+)', self.options)
        if m:
            priority = '<priority>%s</priority>\n' % m.groups()[0]
        solver_options = 'kestrel_options:solver=%s\n' % solver.lower()
        solver_options_key = '%s_options' % solver
        solver_options_value = ''
        if solver_options_key in os.environ:
            solver_options_value = os.getenv(solver_options_key)
        elif solver_options_key.lower() in os.environ:
            solver_options_value = os.getenv(solver_options_key.lower())
        elif solver_options_key.upper() in os.environ:
            solver_options_value = os.getenv(solver_options_key.upper())
        if not solver_options_value == '':
            solver_options += '%s_options:%s\n' % (solver.lower(), solver_options_value)
        nl_string = base64.encodebytes(zipped_nl_file.getvalue()).decode('utf-8')
        xml = '\n              <document>\n              <category>kestrel</category>\n              <email>%s</email>\n              <solver>%s</solver>\n              <inputType>AMPL</inputType>\n              %s\n              <solver_options>%s</solver_options>\n              <nlfile><base64>%s</base64></nlfile>\n' % (self.getEmailAddress(), solver, priority, solver_options, nl_string)
        for key in ampl_files:
            xml += '<%s><![CDATA[%s]]></%s>\n' % (key, ampl_files[key], key)
        for option in ['kestrel_auxfiles', 'mip_priorities', 'objective_precision']:
            if option in os.environ:
                xml += '<%s><![CDATA[%s]]></%s>\n' % (option, os.getenv(option), option)
        xml += '</document>'
        return xml