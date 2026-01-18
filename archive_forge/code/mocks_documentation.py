from boto.mturk.connection import MTurkConnection as RealMTurkConnection

	Mock MTurkConnection that doesn't connect, but instead just prepares
	the request and captures information about its usage.
	