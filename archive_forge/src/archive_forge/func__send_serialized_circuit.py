from typing import List, Optional
import time
import requests
import cirq
import cirq_pasqal
def _send_serialized_circuit(self, serialization_str: str, repetitions: int=1) -> cirq.study.Result:
    """Sends the json string to the remote Pasqal device
        Args:
            serialization_str: Json representation of the circuit.
            repetitions: Number of repetitions.
        Returns:
            json representation of the results
        """
    simulate_url = f'{self.remote_host}/simulate/no-noise/submit'
    submit_response = requests.post(simulate_url, verify=False, headers={'Repetitions': str(repetitions), **self._authorization_header}, data=serialization_str)
    submit_response.raise_for_status()
    task_id = submit_response.text
    result_serialized = self._retrieve_serialized_result(task_id)
    result = cirq.read_json(json_text=result_serialized)
    return result